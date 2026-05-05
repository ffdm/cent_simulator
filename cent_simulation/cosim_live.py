"""Mutable simulator state used by RTL/cocotb co-simulation.

The JSONL trace remains the ordering contract, but this object owns the tensor
state that depends on hardware-produced PNM/RISCV results. Cocotb calls into it
when a hardware result is observed, and it returns the shared-buffer mutations
or expected check values that should now be used for later replay.
"""

import copy
import math

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


def _zero_lanes():
    return ["0000"] * NUM_LANES


def _tensor_to_bf16_hex(tensor):
    return [
        bf16_hex(float_to_bf16_int(value.item()))
        for value in torch.as_tensor(tensor).flatten().to(torch.bfloat16)
    ]


class MutableSingleChannelCentSimulator:
    """Live single-channel simulator state for the hardware co-sim test."""

    def __init__(self):
        self.fixture = get_test_inputs()
        self.red_source_mutations = 0
        self.hardware_rmsnorm_scales = []
        self._self_attention_state = None
        self._final_state = None
        self._live_red_expectations = {}
        self._live_riscv_expectations = {}
        self._red_checks_seen = 0
        self._riscv_checks_seen = 0

    def _scale_float(self, index):
        return bf16_int_to_float(self.hardware_rmsnorm_scales[index])

    def _compute_self_attention_state(self):
        if self._self_attention_state is not None:
            return self._self_attention_state
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
        scores = torch.matmul(xq_for_scores, keys_for_scores) / math.sqrt(head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq_for_scores)
        output = torch.matmul(scores, values_for_output)
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        sa_projection = F.linear(output, dic["wo"].to(torch.float32))
        h = x + sa_projection

        self._self_attention_state = {
            "norm_x": norm_x,
            "xq": xq,
            "xk": xk,
            "xv": xv,
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
        red_scalar, _ = simulate_red_bf16(vectors)
        scale = simulate_rmsnorm_scale_bf16(red_scalar, SINGLE_CHANNEL_DIM)
        return (
            [bf16_hex(red_scalar)] + _zero_lanes()[1:],
            [bf16_hex(scale)] * NUM_LANES,
        )

    def condition_simulate_event(self, event):
        if event.get("op") != "SB_STATE_BEFORE_RED":
            return event

        self.red_source_mutations += 1
        red_index = int(event.get("red_source_index", self.red_source_mutations))
        if not event.get("requires_live_hardware_state"):
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
        expected = None

        if source.startswith("RED hardware output"):
            if red_index == 0:
                self._red_checks_seen += 1
                red_index = self._red_checks_seen
        if source.startswith("RISCV RMSNorm"):
            if rmsnorm_index == 0:
                self._riscv_checks_seen += 1
                rmsnorm_index = self._riscv_checks_seen

        if source.startswith("RED hardware output") and red_index in self._live_red_expectations:
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
        if not event.get("source", "").startswith("RISCV RMSNorm"):
            return
        if not actual_lanes:
            raise AssertionError("RISCV check produced no actual lanes")

        self.hardware_rmsnorm_scales.append(int(actual_lanes[0]))
        if len(self.hardware_rmsnorm_scales) == 1:
            self._compute_self_attention_state()
        elif len(self.hardware_rmsnorm_scales) == 2:
            self._compute_final_state()

    def final_out_tensors(self):
        return self._compute_final_state()["out"], self.fixture["out"]

    def final_out_bf16(self):
        return _tensor_to_bf16_hex(self._compute_final_state()["out"])
