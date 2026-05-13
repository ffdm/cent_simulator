"""Mutable simulator state used by RTL/cocotb co-simulation.

The JSONL trace remains the ordering contract, but this object owns the tensor
state that depends on hardware-produced PNM/RISCV results. Cocotb calls into it
when a hardware result is observed, and it returns the shared-buffer mutations
or expected check values that should now be used for later replay.
"""

import copy
import math
import os

import torch
import torch.nn.functional as F

from generate_cosim_trace import (
    NUM_LANES,
    bf16_hex,
    bf16_int_to_float,
    float_to_bf16_int,
    simulate_red_bf16,
    simulate_rmsnorm_scale_bf16,
)
from test_single_channel import get_test_inputs
from utils import apply_rotary_emb, repeat_kv


SINGLE_CHANNEL_DIM = 4096
SINGLE_CHANNEL_REDUCED_BANKS = 8


def _check_single_channel_geometry(fixture):
    """Validate that the active fixture and constants match the geometry the
    mutable simulator hard-codes.

    The RED source rebuild interleaves NUM_LANES lanes across
    SINGLE_CHANNEL_REDUCED_BANKS banks via ``lane % 2`` (even lanes carry data,
    odd lanes are zero), so it requires ``NUM_LANES == 2 * SINGLE_CHANNEL_REDUCED_BANKS``
    and a model dimension that splits evenly into bank-sized segments. If a
    future model changes ``dim`` or the lane-to-bank mapping, this assertion
    fires immediately instead of silently producing wrong shared-buffer
    mutations downstream.
    """
    if NUM_LANES != 2 * SINGLE_CHANNEL_REDUCED_BANKS:
        raise AssertionError(
            f"cosim_live geometry mismatch: NUM_LANES={NUM_LANES} but the "
            f"interleaved RED rebuild requires NUM_LANES == 2 * "
            f"SINGLE_CHANNEL_REDUCED_BANKS ({2 * SINGLE_CHANNEL_REDUCED_BANKS})"
        )
    if SINGLE_CHANNEL_DIM % SINGLE_CHANNEL_REDUCED_BANKS != 0:
        raise AssertionError(
            f"cosim_live geometry mismatch: SINGLE_CHANNEL_DIM={SINGLE_CHANNEL_DIM} "
            f"is not divisible by SINGLE_CHANNEL_REDUCED_BANKS={SINGLE_CHANNEL_REDUCED_BANKS}"
        )
    fixture_dim = int(fixture["dim"].item())
    if fixture_dim != SINGLE_CHANNEL_DIM:
        raise AssertionError(
            f"cosim_live geometry mismatch: fixture dim={fixture_dim} but "
            f"SINGLE_CHANNEL_DIM={SINGLE_CHANNEL_DIM}; the mutable single-channel "
            f"replay only supports dim={SINGLE_CHANNEL_DIM} today"
        )


def _zero_lanes():
    return ["0000"] * NUM_LANES


def _tensor_to_bf16_hex(tensor):
    return [
        bf16_hex(float_to_bf16_int(value.item()))
        for value in torch.as_tensor(tensor).flatten().to(torch.bfloat16)
    ]


def _softmax_recip_bf16(sum_bf16):
    exp_sum = bf16_int_to_float(sum_bf16)
    return float_to_bf16_int(1.0 / (exp_sum + 1e-10))


def _env_flag_enabled(name):
    return str(os.getenv(name, "0")).strip().lower() in {"1", "true", "yes", "on"}


def _red_accum_f32_enabled():
    return _env_flag_enabled("CENT_ARCH_RED_ACCUM_F32")


def _bf16_add_rne_enabled():
    return _env_flag_enabled("CENT_ARCH_BF16_ADD_RNE")


def _simulate_arch_red_bf16(vectors_bf16):
    if not _red_accum_f32_enabled() and not _bf16_add_rne_enabled():
        red_scalar, _ = simulate_red_bf16(vectors_bf16)
        return red_scalar

    lanes_int = [
        [int(value, 16) if isinstance(value, str) else int(value) for value in lanes]
        for lanes in vectors_bf16
    ]
    try:
        from test_utils import simulate_red_int_rtl
        return simulate_red_int_rtl(
            lanes_int,
            NUM_LANES,
            accum_f32=_red_accum_f32_enabled(),
            add_rne=_bf16_add_rne_enabled(),
            final_rne=_bf16_add_rne_enabled(),
        )
    except ImportError:
        # Fallback for non-cocotb imports. The cocotb path imports the bit-level
        # RTL helper above; this keeps utility imports usable outside tb/.
        accum = torch.tensor(0.0, dtype=torch.float32)
        for lanes in lanes_int:
            vector_sum, _ = simulate_red_bf16([[bf16_hex(value) for value in lanes]])
            accum = (accum + torch.tensor(bf16_int_to_float(vector_sum), dtype=torch.float32)).to(torch.float32)
        return float_to_bf16_int(float(accum.item()))


def _bf16_order_key(value):
    value = int(value) & 0xFFFF
    if value & 0x8000:
        return (~value) & 0xFFFF
    return value | 0x8000


def _bf16_ulp_diff(actual, expected):
    return abs(_bf16_order_key(actual) - _bf16_order_key(expected))


def _tensor_bf16_ints(tensor):
    return [
        float_to_bf16_int(value.item())
        for value in torch.as_tensor(tensor).flatten().to(torch.bfloat16)
    ]


def _accuracy_metrics(actual, expected):
    actual_f32 = torch.as_tensor(actual).flatten().to(torch.float32)
    expected_f32 = torch.as_tensor(expected).flatten().to(torch.float32)
    if actual_f32.numel() != expected_f32.numel():
        raise AssertionError(
            f"accuracy metric shape mismatch: actual={actual_f32.numel()} "
            f"expected={expected_f32.numel()}"
        )
    diff = (actual_f32 - expected_f32).abs()
    rel = diff / torch.clamp(expected_f32.abs(), min=1e-30)
    actual_bf16 = _tensor_bf16_ints(actual_f32)
    expected_bf16 = _tensor_bf16_ints(expected_f32)
    ulps = [
        _bf16_ulp_diff(actual_value, expected_value)
        for actual_value, expected_value in zip(actual_bf16, expected_bf16)
    ]
    sign_crossings = ((actual_f32 < 0) & (expected_f32 > 0)) | ((actual_f32 > 0) & (expected_f32 < 0))
    elements = int(actual_f32.numel())
    return {
        "elements": elements,
        "max_abs": float(diff.max().item()) if elements else 0.0,
        "mean_abs": float(diff.mean().item()) if elements else 0.0,
        "max_rel": float(rel.max().item()) if elements else 0.0,
        "mean_rel": float(rel.mean().item()) if elements else 0.0,
        "max_ulp": int(max(ulps)) if ulps else 0,
        "mean_ulp": float(sum(ulps) / max(len(ulps), 1)),
        "bf16_mismatches": int(sum(1 for actual_value, expected_value in zip(actual_bf16, expected_bf16) if actual_value != expected_value)),
        "sign_crossings": int(sign_crossings.sum().item()) if elements else 0,
    }


def _scalar_bf16_metrics(actual_bf16, expected_bf16):
    return _accuracy_metrics(
        [bf16_int_to_float(actual_bf16)],
        [bf16_int_to_float(expected_bf16)],
    )


def _aggregate_metrics(items):
    if not items:
        return {
            "elements": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
            "mean_rel": 0.0,
            "max_ulp": 0,
            "mean_ulp": 0.0,
            "bf16_mismatches": 0,
            "sign_crossings": 0,
        }
    elements = sum(int(item.get("elements", 0)) for item in items)
    return {
        "elements": int(elements),
        "max_abs": max(float(item.get("max_abs", 0.0)) for item in items),
        "mean_abs": float(sum(float(item.get("mean_abs", 0.0)) * int(item.get("elements", 0)) for item in items) / max(elements, 1)),
        "max_rel": max(float(item.get("max_rel", 0.0)) for item in items),
        "mean_rel": float(sum(float(item.get("mean_rel", 0.0)) * int(item.get("elements", 0)) for item in items) / max(elements, 1)),
        "max_ulp": max(int(item.get("max_ulp", 0)) for item in items),
        "mean_ulp": float(sum(float(item.get("mean_ulp", 0.0)) * int(item.get("elements", 0)) for item in items) / max(elements, 1)),
        "bf16_mismatches": int(sum(int(item.get("bf16_mismatches", 0)) for item in items)),
        "sign_crossings": int(sum(int(item.get("sign_crossings", 0)) for item in items)),
    }


def _tensor_stats(tensor):
    values = torch.as_tensor(tensor).flatten().to(torch.float32)
    elements = int(values.numel())
    if elements == 0:
        return {
            "elements": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "negative": 0,
            "zero": 0,
            "positive": 0,
        }
    return {
        "elements": elements,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "negative": int((values < 0).sum().item()),
        "zero": int((values == 0).sum().item()),
        "positive": int((values > 0).sum().item()),
    }


class MutableSingleChannelCentSimulator:
    """Live single-channel simulator state for the hardware co-sim test."""

    def __init__(self):
        self.context_len = int(os.getenv("SINGLE_CHANNEL_COSIM_CONTEXT_LEN", "1"))
        self.softmax_impl = os.getenv("SINGLE_CHANNEL_COSIM_SOFTMAX_IMPL", "python")
        self.softmax_centering = os.getenv("SINGLE_CHANNEL_COSIM_SOFTMAX_CENTERING", "regular")
        if self.softmax_centering not in {"regular", "subtract-max"}:
            raise ValueError(
                "SINGLE_CHANNEL_COSIM_SOFTMAX_CENTERING must be 'regular' or "
                f"'subtract-max', got {self.softmax_centering!r}"
            )
        self.fixture = get_test_inputs(context_len=self.context_len)
        self.fixture_mode = self.fixture.get("fixture_mode", "legacy")
        _check_single_channel_geometry(self.fixture)
        self.red_source_mutations = 0
        self.hardware_rmsnorm_scales = []
        self.hardware_rmsnorm_red = []
        self.hardware_softmax_exp = {}
        self.hardware_softmax_red = {}
        self.hardware_softmax_recip = {}
        self.softmax_exp_metrics = {}
        self.rmsnorm_red_metrics = []
        self.rmsnorm_scale_metrics = []
        self._attention_pre_softmax_state = None
        self._self_attention_state = None
        self._final_state = None
        self._live_red_expectations = {}
        self._live_riscv_expectations = {}
        self._red_checks_seen = 0
        self._riscv_checks_seen = 0

    def _scale_float(self, index):
        return bf16_int_to_float(self.hardware_rmsnorm_scales[index])

    def _softmax_exp_input(self, pre_scores, head):
        scores = pre_scores[0, head, 0].flatten().to(torch.float32)
        if self.softmax_centering == "subtract-max":
            scores = scores - scores.max()
        return scores

    def _compute_attention_pre_softmax_state(self):
        if self._attention_pre_softmax_state is not None:
            return self._attention_pre_softmax_state
        if len(self.hardware_rmsnorm_scales) < 1:
            raise AssertionError(
                "mutable simulator needs the first hardware RISCV RMSNorm scale "
                "before it can rebuild self_attention_aim-dependent PIM state"
            )

        dic = self.fixture
        dim = int(dic["dim"].item())
        n_heads = int(dic["n_heads"].item())
        head_dim = dim // n_heads
        bsz, seqlen, _ = dic["x"].shape
        start_pos = int(dic["start_pos"].item())

        x = dic["x"].to(torch.float32)
        norm_x = x * self._scale_float(0) * dic["SANorm"].to(torch.float32)
        xq = F.linear(norm_x, dic["wq"].to(torch.float32))
        xk = F.linear(norm_x, dic["wk"].to(torch.float32))
        xv = F.linear(norm_x, dic["wv"].to(torch.float32))

        xq_heads = xq.reshape(bsz, seqlen, n_heads, head_dim)
        xk_heads = xk.reshape(bsz, seqlen, n_heads, head_dim)
        xv_heads = xv.reshape(bsz, seqlen, n_heads, head_dim)
        xq_rot, xk_rot = apply_rotary_emb(xq_heads, xk_heads, dic["freqs_cis"])

        cache_k = dic["cache_k"].clone().to(torch.float32)
        cache_v = dic["cache_v"].clone().to(torch.float32)
        cache_k[:bsz, start_pos:start_pos + seqlen] = xk_rot
        cache_v[:bsz, start_pos:start_pos + seqlen] = xv_heads
        keys = cache_k[:bsz, :start_pos + seqlen]
        values = cache_v[:bsz, :start_pos + seqlen]
        if "n_kv_heads" in dic:
            n_repeat = n_heads // int(dic["n_kv_heads"].item())
            keys = repeat_kv(keys, n_repeat)
            values = repeat_kv(values, n_repeat)

        xq_for_scores = xq_rot.transpose(1, 2)
        keys_for_scores = keys.transpose(1, 2).transpose(2, 3)
        values_for_output = values.transpose(1, 2)
        pre_softmax_scores = torch.matmul(xq_for_scores, keys_for_scores) / math.sqrt(head_dim)

        self._attention_pre_softmax_state = {
            "norm_x": norm_x,
            "xq": xq,
            "xk": xk,
            "xv": xv,
            "pre_softmax_scores": pre_softmax_scores,
            "values_for_output": values_for_output,
        }
        return self._attention_pre_softmax_state

    def _hardware_softmax_scores(self, template):
        dic = self.fixture
        n_heads = int(dic["n_heads"].item())
        if len(self.hardware_softmax_exp) != n_heads:
            raise AssertionError(
                "mutable simulator needs all hardware EXP softmax outputs before "
                f"building the attention output: saw {len(self.hardware_softmax_exp)}/{n_heads}"
            )
        if len(self.hardware_softmax_recip) != n_heads:
            raise AssertionError(
                "mutable simulator needs all hardware softmax reciprocals before "
                f"building the attention output: saw {len(self.hardware_softmax_recip)}/{n_heads}"
            )

        heads = []
        for head in range(n_heads):
            exp_lanes = [
                lane
                for chunk in self.hardware_softmax_exp[head]
                for lane in chunk
            ][:self.context_len]
            exp_values = torch.tensor(
                [bf16_int_to_float(lane) for lane in exp_lanes],
                dtype=torch.float32,
            )
            recip = torch.tensor(
                bf16_int_to_float(self.hardware_softmax_recip[head]),
                dtype=torch.float32,
            )
            heads.append((exp_values.to(torch.bfloat16) * recip.to(torch.bfloat16)).to(torch.float32))
        return torch.stack(heads).reshape_as(template)

    def _compute_self_attention_state(self):
        if self._self_attention_state is not None:
            return self._self_attention_state

        pre_state = self._compute_attention_pre_softmax_state()
        x = self.fixture["x"].to(torch.float32)
        values_for_output = pre_state["values_for_output"]
        pre_softmax_scores = pre_state["pre_softmax_scores"]

        if self.softmax_impl == "pnm-functional":
            scores = self._hardware_softmax_scores(pre_softmax_scores)
        else:
            scores = F.softmax(pre_softmax_scores, dim=-1).type_as(pre_softmax_scores)
        output = torch.matmul(scores, values_for_output)
        bsz, seqlen, _ = x.shape
        dim = int(self.fixture["dim"].item())
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        sa_projection = F.linear(output, self.fixture["wo"].to(torch.float32))
        h = x + sa_projection

        self._self_attention_state = {
            "norm_x": pre_state["norm_x"],
            "xq": pre_state["xq"],
            "xk": pre_state["xk"],
            "xv": pre_state["xv"],
            "pre_softmax_scores": pre_softmax_scores,
            "scores": scores,
            "output": output,
            "sa": sa_projection,
            "h": h,
        }
        return self._self_attention_state

    def _compute_final_state(self):
        if self._final_state is not None:
            return self._final_state
        if len(self.hardware_rmsnorm_scales) < 2:
            raise AssertionError(
                "mutable simulator needs both hardware RISCV RMSNorm scales "
                "before it can compute the final FFN output"
            )

        dic = self.fixture
        h = self._compute_self_attention_state()["h"]
        norm_h = h * self._scale_float(1) * dic["FFNNorm"].to(torch.float32)
        x1 = F.linear(norm_h, dic["w1"].to(torch.float32))
        x3 = F.linear(norm_h, dic["w3"].to(torch.float32))
        ffn = F.linear(F.silu(x1) * x3, dic["w2"].to(torch.float32))
        out = h + ffn
        self._final_state = {
            "norm_h": norm_h,
            "x1": x1,
            "x3": x3,
            "ffn": ffn,
            "out": out,
        }
        return self._final_state

    def _red_source_writes_from_tensor(self, template_event, tensor, description):
        writes = template_event.get("writes", [])
        if len(writes) != NUM_LANES:
            raise AssertionError(
                f"single-channel RED source mutation should have {NUM_LANES} writes, "
                f"got {len(writes)} at line {template_event.get('_line')}"
            )

        flat = torch.as_tensor(tensor).flatten().to(torch.bfloat16)
        if flat.numel() != SINGLE_CHANNEL_DIM:
            raise AssertionError(
                f"single-channel RED source tensor should have {SINGLE_CHANNEL_DIM} elements, "
                f"got {flat.numel()}"
            )

        segment_width = SINGLE_CHANNEL_DIM // SINGLE_CHANNEL_REDUCED_BANKS
        first_lanes = []
        for lane in range(NUM_LANES):
            if lane % 2:
                first_lanes.append("0000")
                continue
            bank_group = lane // 2
            start = bank_group * segment_width
            stop = start + segment_width
            bank_sum = (flat[start:stop] * flat[start:stop]).to(torch.bfloat16).sum()
            first_lanes.append(bf16_hex(float_to_bf16_int(bank_sum.item())))

        live_writes = []
        for index, write in enumerate(writes):
            updated = dict(write)
            updated["lanes_bf16"] = first_lanes if index == 0 else _zero_lanes()
            updated["description"] = description
            live_writes.append(updated)
        return live_writes

    def _compute_red_and_riscv_expectations(self, writes):
        vectors = [write["lanes_bf16"] for write in writes]
        red_scalar = _simulate_arch_red_bf16(vectors)
        scale = simulate_rmsnorm_scale_bf16(red_scalar, SINGLE_CHANNEL_DIM)
        return (
            [bf16_hex(red_scalar)] + _zero_lanes()[1:],
            [bf16_hex(scale)] * NUM_LANES,
        )

    def condition_simulate_event(self, event):
        if event.get("op") == "SB_STATE_BEFORE_EXP" and event.get("requires_live_hardware_state"):
            head = int(event.get("softmax_head_index", 0))
            pre_scores = self._compute_attention_pre_softmax_state()["pre_softmax_scores"]
            flat = self._softmax_exp_input(pre_scores, head).to(torch.bfloat16)
            if flat.numel() % NUM_LANES != 0:
                raise AssertionError(
                    "hardware softmax EXP input must be an exact multiple of "
                    f"{NUM_LANES} lanes; got {flat.numel()}"
                )
            chunks = [
                _tensor_to_bf16_hex(flat[index:index + NUM_LANES])
                for index in range(0, flat.numel(), NUM_LANES)
            ]
            writes = event.get("writes", [])
            if len(writes) != len(chunks):
                raise AssertionError(
                    f"softmax head {head} expected {len(chunks)} EXP source chunks, "
                    f"contract has {len(writes)} writes"
                )
            live_writes = []
            for write, lanes in zip(writes, chunks):
                updated = dict(write)
                updated["lanes_bf16"] = lanes
                updated["description"] = "mutable cent_simulator hardware-RMS-conditioned softmax EXP input"
                live_writes.append(updated)

            conditioned = copy.deepcopy(event)
            conditioned["writes"] = live_writes
            conditioned["source"] = "mutable_cent_simulator.self_attention_aim.softmax_EXP_inputs"
            conditioned["live_conditioned"] = True
            conditioned["live_dependency_satisfied"] = "hardware_rmsnorm_scale[0]"
            conditioned["live_scale_bf16"] = bf16_hex(self.hardware_rmsnorm_scales[0])
            conditioned["live_lane_diffs_vs_template"] = sum(
                1
                for original, live in zip(event.get("writes", []), live_writes)
                for lhs, rhs in zip(original.get("lanes_bf16", []), live.get("lanes_bf16", []))
                if lhs.lower() != rhs.lower()
            )
            return conditioned

        if event.get("op") != "SB_STATE_BEFORE_RED":
            return event

        self.red_source_mutations += 1
        red_index = int(event.get("red_source_index", self.red_source_mutations))
        if not event.get("requires_live_hardware_state"):
            red_lanes, riscv_lanes = self._compute_red_and_riscv_expectations(event.get("writes", []))
            self._live_red_expectations[red_index] = red_lanes
            self._live_riscv_expectations[red_index] = riscv_lanes
            return event

        if red_index != 2:
            return event

        h = self._compute_self_attention_state()["h"]
        live_writes = self._red_source_writes_from_tensor(
            event,
            h,
            "mutable cent_simulator self_attention_aim output before FFN RED",
        )
        red_lanes, riscv_lanes = self._compute_red_and_riscv_expectations(live_writes)
        self._live_red_expectations[red_index] = red_lanes
        self._live_riscv_expectations[red_index] = riscv_lanes

        conditioned = copy.deepcopy(event)
        conditioned["writes"] = live_writes
        conditioned["source"] = "mutable_cent_simulator.self_attention_aim.RED_inputs"
        conditioned["live_conditioned"] = True
        conditioned["live_dependency_satisfied"] = "hardware_rmsnorm_scale[0]"
        conditioned["live_scale_bf16"] = bf16_hex(self.hardware_rmsnorm_scales[0])
        conditioned["live_lane_diffs_vs_template"] = sum(
            1
            for original, live in zip(event.get("writes", []), live_writes)
            for lhs, rhs in zip(original.get("lanes_bf16", []), live.get("lanes_bf16", []))
            if lhs.lower() != rhs.lower()
        )
        return conditioned

    def condition_check_event(self, event):
        source = event.get("source", "")
        red_index = int(event.get("red_source_index", 0))
        rmsnorm_index = int(event.get("rmsnorm_index", red_index))
        softmax_head = event.get("softmax_head_index")
        expected = None

        if source.startswith("RED hardware output"):
            if red_index == 0:
                self._red_checks_seen += 1
                red_index = self._red_checks_seen
        if source.startswith("RISCV RMSNorm"):
            if rmsnorm_index == 0:
                self._riscv_checks_seen += 1
                rmsnorm_index = self._riscv_checks_seen

        if source.startswith("RED Softmax") and softmax_head is not None:
            softmax_head = int(softmax_head)
            if softmax_head in self.hardware_softmax_exp:
                vectors = [
                    [bf16_hex(lane) for lane in chunk]
                    for chunk in self.hardware_softmax_exp[softmax_head]
                ]
                red_scalar = _simulate_arch_red_bf16(vectors)
                expected = [bf16_hex(red_scalar)] + _zero_lanes()[1:]
        elif source.startswith("RISCV Softmax") and softmax_head is not None:
            softmax_head = int(softmax_head)
            if softmax_head in self.hardware_softmax_red:
                recip = _softmax_recip_bf16(self.hardware_softmax_red[softmax_head])
                expected = [bf16_hex(recip)] * NUM_LANES
        elif source.startswith("RED hardware output") and red_index in self._live_red_expectations:
            expected = self._live_red_expectations[red_index]
        elif source.startswith("RISCV RMSNorm") and rmsnorm_index in self._live_riscv_expectations:
            expected = self._live_riscv_expectations[rmsnorm_index]

        if expected is None:
            return event

        conditioned = copy.deepcopy(event)
        conditioned["expected_lanes_bf16"] = expected
        conditioned["live_conditioned"] = True
        conditioned["source"] = source + " (mutable cent_simulator expected)"
        return conditioned

    def observe_check_event(self, event, actual_lanes):
        source = event.get("source", "")
        if source.startswith("RED Softmax"):
            head = int(event["softmax_head_index"])
            self.hardware_softmax_red[head] = int(actual_lanes[0])
            return
        if source.startswith("RED hardware output"):
            if not actual_lanes:
                raise AssertionError("RED check produced no actual lanes")
            self.hardware_rmsnorm_red.append(int(actual_lanes[0]))
            expected = int(event["expected_lanes_bf16"][0], 16)
            self.rmsnorm_red_metrics.append(_scalar_bf16_metrics(int(actual_lanes[0]), expected))
            return
        if source.startswith("RISCV Softmax"):
            head = int(event["softmax_head_index"])
            self.hardware_softmax_recip[head] = int(actual_lanes[0])
            self._self_attention_state = None
            self._final_state = None
            return
        if not source.startswith("RISCV RMSNorm"):
            return
        if not actual_lanes:
            raise AssertionError("RISCV check produced no actual lanes")

        self.hardware_rmsnorm_scales.append(int(actual_lanes[0]))
        expected = int(event["expected_lanes_bf16"][0], 16)
        self.rmsnorm_scale_metrics.append(_scalar_bf16_metrics(int(actual_lanes[0]), expected))
        if len(self.hardware_rmsnorm_scales) == 1:
            self._compute_attention_pre_softmax_state()
            if self.softmax_impl != "pnm-functional":
                self._compute_self_attention_state()
        elif len(self.hardware_rmsnorm_scales) == 2:
            self._compute_final_state()

    def observe_softmax_exp(self, event, actual_chunks):
        from test_utils import (
            bfloat16_int_to_float as tb_bf16_to_float,
            float_to_bfloat16_int as tb_float_to_bf16,
            simulate_exp_taylor_rtl,
        )

        head = int(event["softmax_head_index"])
        actual = [
            [int(lane) for lane in chunk]
            for chunk in actual_chunks
        ]
        pre_scores = self._compute_attention_pre_softmax_state()["pre_softmax_scores"]
        exp_input = self._softmax_exp_input(pre_scores, head)
        exp_source = exp_input.to(torch.bfloat16).to(torch.float32)
        chunks = [
            exp_source[index:index + NUM_LANES]
            for index in range(0, exp_source.numel(), NUM_LANES)
        ]
        expected = [
            simulate_exp_taylor_rtl([float(value.item()) for value in chunk])
            for chunk in chunks
        ]
        expected_flat = [lane for chunk in expected for lane in chunk][:self.context_len]
        ulp_diffs = [
            _bf16_ulp_diff(act_lane, exp_lane)
            for act_lane, exp_lane in zip(
                [lane for chunk in actual for lane in chunk][:self.context_len],
                expected_flat,
            )
        ]
        max_ulp = max(ulp_diffs) if ulp_diffs else 0

        actual_flat = [lane for chunk in actual for lane in chunk][:self.context_len]
        actual_values = torch.tensor(
            [tb_bf16_to_float(lane) for lane in actual_flat],
            dtype=torch.float32,
        )
        torch_exp = torch.exp(exp_source).to(torch.bfloat16)
        torch_exp_values = torch_exp.to(torch.float32)
        torch_exp_bf16 = [tb_float_to_bf16(float(value.item())) for value in torch_exp]
        abs_diff = (actual_values - torch_exp_values).abs()
        rel_diff = abs_diff / torch.clamp(torch_exp_values.abs(), min=1e-30)
        expected_values = torch.tensor(
            [tb_bf16_to_float(lane) for lane in expected_flat],
            dtype=torch.float32,
        )
        rtl_diff = (actual_values - expected_values).abs()
        rtl_rel = rtl_diff / torch.clamp(expected_values.abs(), min=1e-30)
        rtl_sign_crossings = ((actual_values < 0) & (expected_values > 0)) | (
            (actual_values > 0) & (expected_values < 0)
        )
        torch_sign_crossings = ((actual_values < 0) & (torch_exp_values > 0)) | (
            (actual_values > 0) & (torch_exp_values < 0)
        )
        torch_ulp_diffs = [
            _bf16_ulp_diff(actual_lane, torch_lane)
            for actual_lane, torch_lane in zip(actual_flat, torch_exp_bf16)
        ]
        self.softmax_exp_metrics[head] = {
            "max_ulp_vs_rtl_taylor": int(max_ulp),
            "max_ulp_vs_rtl_taylor_exceeds_2": bool(max_ulp > 2),
            "mean_ulp_vs_rtl_taylor": float(sum(ulp_diffs) / max(len(ulp_diffs), 1)),
            "max_abs_vs_rtl_taylor": float(rtl_diff.max().item()),
            "mean_abs_vs_rtl_taylor": float(rtl_diff.mean().item()),
            "max_rel_vs_rtl_taylor": float(rtl_rel.max().item()),
            "mean_rel_vs_rtl_taylor": float(rtl_rel.mean().item()),
            "sign_crossings_vs_rtl_taylor": int(rtl_sign_crossings.sum().item()),
            "max_abs_vs_torch_exp_bf16": float(abs_diff.max().item()),
            "mean_abs_vs_torch_exp_bf16": float(abs_diff.mean().item()),
            "max_rel_vs_torch_exp_bf16": float(rel_diff.max().item()),
            "mean_rel_vs_torch_exp_bf16": float(rel_diff.mean().item()),
            "max_ulp_vs_torch_exp_bf16": int(max(torch_ulp_diffs)) if torch_ulp_diffs else 0,
            "mean_ulp_vs_torch_exp_bf16": float(sum(torch_ulp_diffs) / max(len(torch_ulp_diffs), 1)),
            "bf16_mismatches_vs_torch_exp": int(sum(
                1 for actual_lane, torch_lane in zip(actual_flat, torch_exp_bf16)
                if int(actual_lane) != int(torch_lane)
            )),
            "sign_crossings_vs_torch_exp": int(torch_sign_crossings.sum().item()),
        }
        self.hardware_softmax_exp[head] = actual
        self._self_attention_state = None
        self._final_state = None

    def accuracy_report(self):
        pre_scores = self._compute_attention_pre_softmax_state()["pre_softmax_scores"]
        pytorch_softmax = F.softmax(pre_scores, dim=-1).type_as(pre_scores)
        hardware_softmax = self._hardware_softmax_scores(pre_scores)
        softmax_metrics = _accuracy_metrics(hardware_softmax, pytorch_softmax)
        n_heads = int(self.fixture["n_heads"].item())
        exp_inputs = torch.stack([
            self._softmax_exp_input(pre_scores, head)
            for head in range(n_heads)
        ])

        out_hw, golden = self.final_out_tensors()
        final_metrics = _accuracy_metrics(out_hw, golden)
        hw_bf16 = self.final_out_bf16()
        golden_bf16 = _tensor_to_bf16_hex(golden)

        exp_metrics = list(self.softmax_exp_metrics.values())
        return {
            "context_len": self.context_len,
            "fixture_mode": self.fixture_mode,
            "softmax_impl": self.softmax_impl,
            "softmax_centering": self.softmax_centering,
            "red_accum_f32": bool(_red_accum_f32_enabled()),
            "bf16_add_rne": bool(_bf16_add_rne_enabled()),
            "rmsnorm_scales_bf16": [bf16_hex(value) for value in self.hardware_rmsnorm_scales],
            "softmax_heads_observed": len(self.hardware_softmax_exp),
            "softmax_raw_score_stats": _tensor_stats(pre_scores),
            "softmax_exp_input_stats": _tensor_stats(exp_inputs),
            "softmax_probability_stats": _tensor_stats(hardware_softmax),
            "softmax_exp_vs_rtl_taylor": {
                "max_ulp": max((item["max_ulp_vs_rtl_taylor"] for item in exp_metrics), default=0),
                "heads_exceeding_2ulp": int(sum(
                    1 for item in exp_metrics if item["max_ulp_vs_rtl_taylor_exceeds_2"]
                )),
                "mean_ulp_mean_over_heads": float(sum(
                    item["mean_ulp_vs_rtl_taylor"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "max_abs": max((item["max_abs_vs_rtl_taylor"] for item in exp_metrics), default=0.0),
                "mean_abs_mean_over_heads": float(sum(
                    item["mean_abs_vs_rtl_taylor"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "max_rel": max((item["max_rel_vs_rtl_taylor"] for item in exp_metrics), default=0.0),
                "mean_rel_mean_over_heads": float(sum(
                    item["mean_rel_vs_rtl_taylor"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "sign_crossings_total": int(sum(
                    item["sign_crossings_vs_rtl_taylor"] for item in exp_metrics
                )),
                "elements": int(self.context_len * len(exp_metrics)),
            },
            "softmax_exp_vs_torch_exp_bf16": {
                "max_abs": max((item["max_abs_vs_torch_exp_bf16"] for item in exp_metrics), default=0.0),
                "mean_abs_mean_over_heads": float(sum(
                    item["mean_abs_vs_torch_exp_bf16"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "max_rel": max((item["max_rel_vs_torch_exp_bf16"] for item in exp_metrics), default=0.0),
                "mean_rel_mean_over_heads": float(sum(
                    item["mean_rel_vs_torch_exp_bf16"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "max_ulp": max((item["max_ulp_vs_torch_exp_bf16"] for item in exp_metrics), default=0),
                "mean_ulp_mean_over_heads": float(sum(
                    item["mean_ulp_vs_torch_exp_bf16"] for item in exp_metrics
                ) / max(len(exp_metrics), 1)),
                "bf16_mismatches_total": int(sum(
                    item["bf16_mismatches_vs_torch_exp"] for item in exp_metrics
                )),
                "sign_crossings_total": int(sum(
                    item["sign_crossings_vs_torch_exp"] for item in exp_metrics
                )),
                "elements": int(self.context_len * len(exp_metrics)),
            },
            "softmax_probability_vs_pytorch": softmax_metrics,
            "final_out_vs_pytorch": {
                **final_metrics,
                "bf16_mismatches": int(sum(1 for a, b in zip(hw_bf16, golden_bf16) if a != b)),
            },
            "rmsnorm_red_vs_expected": _aggregate_metrics(self.rmsnorm_red_metrics),
            "rmsnorm_scale_vs_expected": _aggregate_metrics(self.rmsnorm_scale_metrics),
            "softmax_exp_per_head": self.softmax_exp_metrics,
        }

    def final_out_tensors(self):
        return self._compute_final_state()["out"], self.fixture["out"]

    def final_out_bf16(self):
        return _tensor_to_bf16_hex(self._compute_final_state()["out"])
