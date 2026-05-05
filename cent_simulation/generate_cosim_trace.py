#!/usr/bin/env python3
"""Generate JSONL contracts for CENT simulator/RTL co-simulation.

The JSONL stream is intentionally executable by cocotb without importing the
simulator.  Unsupported PIM work is represented as explicit shared-buffer
mutations; hardware-supported RED, ACC, and RISCV operations are represented as
encoded decoder instructions.
"""

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from pnm_sim import PC_RMSNORM_SCALE

NUM_LANES = 16
ISA_SELECTOR_WIDTH = 6
ISA_OPSIZE_WIDTH = 10
OPSIZE_DATAPATH_WIDTH = 16
RISCV_DIM_WIDTH = 20
REQUEST_WIDTH = 64
OPSIZE_MAX = (1 << ISA_OPSIZE_WIDTH) - 1
INSTR_HEX_WIDTH = (REQUEST_WIDTH + 3) // 4
SCHEMA = "cent-cosim-jsonl"
SCHEMA_VERSION = 1

MASK_SEL_ZERO = 0
MASK_SEL_ALL = 1
MASK_SEL_ONEHOT_BASE = 2
MASK_SEL_CTX_BASE = 34
MASK_CTX_ENTRIES = 30

RISCV = 16
ISR_SYNC = 17
ISR_EOC = 18
SET_MASK_CTX = 19
SET_CHMASK = SET_MASK_CTX
EXP = 20
RED = 21
ACC = 22
SET_RISCV_PC = 23
SET_RISCV_CTX = SET_RISCV_PC

PIM_OPCODES = {
    "WR_SBK": 1,
    "WR_GB": 2,
    "WR_BIAS": 3,
    "WR_AFLUT": 4,
    "RD_MAC": 5,
    "RD_AF": 6,
    "RD_SBK": 7,
    "COPY_BKGB": 8,
    "COPY_GBBK": 9,
    "MAC_SBK": 10,
    "MAC_ABK": 11,
    "AF": 12,
    "EWMUL": 13,
    "EWADD": 14,
    "WR_ABK": 15,
    "SYNC": 17,
    "EOC": 18,
}

OPCODES = {
    "RISCV": RISCV,
    "ISR_SYNC": ISR_SYNC,
    "ISR_EOC": ISR_EOC,
    "SET_MASK_CTX": SET_MASK_CTX,
    "SET_CHMASK": SET_CHMASK,
    "RED": RED,
    "ACC": ACC,
    "SET_RISCV_PC": SET_RISCV_PC,
    "SET_RISCV_CTX": SET_RISCV_CTX,
}
HARDWARE_SUPPORTED_OPS = ["RED", "ACC", "RISCV"]


def float_to_bf16_int(value):
    packed = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    return packed >> 16


def bf16_int_to_float(value):
    packed = int(value) << 16
    return struct.unpack(">f", struct.pack(">I", packed))[0]


def bf16_add_int(a_value, b_value):
    return float_to_bf16_int(bf16_int_to_float(a_value) + bf16_int_to_float(b_value))


def bf16_hex(value):
    return f"{int(value) & 0xFFFF:04x}"


def zero_lanes():
    return ["0000"] * NUM_LANES


def tensor_to_bf16_hex(values):
    tensor = torch.as_tensor(values).flatten().to(torch.bfloat16)
    return [bf16_hex(float_to_bf16_int(v.item())) for v in tensor]


def pack_chunks(values, chunk_size=NUM_LANES):
    values = torch.as_tensor(values).flatten()
    assert values.numel() % chunk_size == 0
    return [
        tensor_to_bf16_hex(values[index:index + chunk_size])
        for index in range(0, values.numel(), chunk_size)
    ]


def simulate_adder_tree_bf16(lanes_bf16):
    current = [int(value, 16) if isinstance(value, str) else int(value) for value in lanes_bf16]
    assert len(current) == NUM_LANES
    while len(current) > 1:
        current = [
            bf16_add_int(current[2 * index], current[2 * index + 1])
            for index in range(len(current) // 2)
        ]
    return current[0]


def simulate_red_bf16(vectors_bf16):
    accum = 0
    steps = []
    for index, lanes in enumerate(vectors_bf16):
        vector_sum = simulate_adder_tree_bf16(lanes)
        prev = accum
        accum = bf16_add_int(accum, vector_sum)
        steps.append({
            "index": index,
            "vector_sum_bf16": bf16_hex(vector_sum),
            "prev_bf16": bf16_hex(prev),
            "accum_bf16": bf16_hex(accum),
        })
    return accum, steps


def simulate_rmsnorm_scale_bf16(red_scalar_bf16, dim):
    sum_f32 = np.float32(bf16_int_to_float(red_scalar_bf16))
    dim_f32 = np.float32(dim)
    eps_f32 = np.float32(1e-5)
    one_f32 = np.float32(1.0)
    mean_f32 = sum_f32 / dim_f32
    inner_f32 = mean_f32 + eps_f32
    root_f32 = np.sqrt(inner_f32).astype(np.float32)
    scale_f32 = one_f32 / root_f32
    return float_to_bf16_int(scale_f32)


def _check_selector(selector, label="selector"):
    if not 0 <= selector < (1 << ISA_SELECTOR_WIDTH):
        raise ValueError(f"{label} must fit in {ISA_SELECTOR_WIDTH} bits: {selector}")


def _check_mask_ctx_id(ctx_id):
    if not 0 <= ctx_id < MASK_CTX_ENTRIES:
        raise ValueError(f"mask context id must be 0..{MASK_CTX_ENTRIES - 1}: {ctx_id}")


def _mask_to_direct_selector(chmask):
    mask = int(chmask) & 0xFFFFFFFF
    if mask == 0:
        return MASK_SEL_ZERO
    if mask == 0xFFFFFFFF:
        return MASK_SEL_ALL
    if mask & (mask - 1) == 0:
        return MASK_SEL_ONEHOT_BASE + (mask.bit_length() - 1)
    return None


def build_set_chmask(ctx_id, chmask):
    _check_mask_ctx_id(ctx_id)
    instr = 0
    instr |= (SET_MASK_CTX & 0x1F) << 59
    instr |= (ctx_id & 0x3F) << 53
    instr |= (int(chmask) & 0xFFFFFFFF) << 21
    return instr


def build_set_riscv_pc(pc_id, pc):
    _check_selector(pc_id, "pc_id")
    instr = 0
    instr |= (SET_RISCV_PC & 0x1F) << 59
    instr |= (pc_id & 0x3F) << 53
    instr |= (int(pc) & 0xFFFFFFFF) << 21
    return instr


def build_set_riscv_ctx(ctx_id, pc, dim):
    del dim
    return build_set_riscv_pc(ctx_id, pc)


def build_raw_instruction(cmd, ch_high=0, ch_low=0, chmask_id=0, mask_sel=None, opsize=0, bank=0, ro=0, co=0, r1=0, flags=0):
    del ch_high, ch_low
    if not 0 <= opsize <= OPSIZE_MAX:
        raise ValueError(f"opsize must fit in {ISA_OPSIZE_WIDTH} bits: {opsize}")
    if mask_sel is None:
        mask_sel = chmask_id
    _check_selector(mask_sel, "mask_sel")
    instr = 0
    instr |= (cmd & 0x1F) << 59
    instr |= (mask_sel & 0x3F) << 53
    instr |= (opsize & OPSIZE_MAX) << 43
    instr |= (bank & 0xF) << 39
    instr |= (ro & 0x3FFF) << 25
    instr |= (co & 0x3FF) << 15
    instr |= (r1 & 0x7FF) << 4
    instr |= (flags & 0xF)
    return instr


def _check_riscv_dim(dim):
    if not 0 <= dim < (1 << RISCV_DIM_WIDTH):
        raise ValueError(f"RISCV dim must fit in {RISCV_DIM_WIDTH} bits: {dim}")


def build_instruction(cmd, opsize=0, r0=0, r1=0, pc=0, dim=0, ctx_id=0, chmask_id=0, pc_id=None, sub=0, rs2=0):
    del pc, chmask_id
    if not 0 <= opsize <= OPSIZE_MAX:
        raise ValueError(f"opsize must fit in {ISA_OPSIZE_WIDTH} bits: {opsize}")
    instr = 0
    instr |= (cmd & 0x1F) << 59
    if cmd == RISCV:
        _check_selector(pc_id if pc_id is not None else ctx_id, "pc_id")
        _check_riscv_dim(dim)
        instr |= ((pc_id if pc_id is not None else ctx_id) & 0x3F) << 53
        instr |= (opsize & OPSIZE_MAX) << 43
        instr |= (r0 & 0x7FF) << 32
        instr |= (r1 & 0x7FF) << 21
        instr |= (int(dim) & ((1 << RISCV_DIM_WIDTH) - 1)) << 1
    elif cmd in {EXP, RED, ACC}:
        _check_selector(sub, "sub")
        instr |= (sub & 0x3F) << 53
        instr |= (opsize & OPSIZE_MAX) << 43
        instr |= (r0 & 0x7FF) << 32
        instr |= (r1 & 0x7FF) << 21
        instr |= (rs2 & 0x7FF) << 10
    else:
        instr |= (opsize & OPSIZE_MAX) << 43
    return instr


def instruction_event(op, opsize=0, rd=0, rs=0, pc=0, dim=0, pc_id=0, case=None, note=None):
    cmd = OPCODES[op]
    instr = build_instruction(cmd, opsize=opsize, r0=rd, r1=rs, pc=pc, dim=dim, pc_id=pc_id)
    event = {
        "type": "instruction",
        "op": op,
        "execute": "control" if op == "ISR_EOC" else "hardware",
        "encoded": f"0x{instr:0{INSTR_HEX_WIDTH}x}",
        "fields": {
            "cmd": cmd,
            "opsize": opsize,
            "r0": rd,
            "r1": rs,
        },
    }
    if op == "RISCV":
        event["fields"]["pc_id"] = pc_id
        event["fields"]["pc"] = f"0x{pc:08x}"
        event["fields"]["dim"] = dim
    if op in {"SET_MASK_CTX", "SET_CHMASK", "SET_RISCV_PC", "SET_RISCV_CTX"}:
        event["execute"] = "context"
    if case is not None:
        event["case"] = case
    if op == "ACC":
        event["semantics"] = "SB[r0 + i] = SB[r0 + i] + SB[r1 + i]"
    if note:
        event["note"] = note
    return event


def set_chmask_event(ctx_id, chmask):
    instr = build_set_chmask(ctx_id, chmask)
    return {
        "type": "instruction",
        "op": "SET_MASK_CTX",
        "execute": "context",
        "encoded": _hex_instruction(instr),
        "fields": {
            "cmd": SET_MASK_CTX,
            "mask_ctx_id": ctx_id,
            "chmask": f"0x{int(chmask) & 0xFFFFFFFF:08x}",
        },
    }


def set_riscv_ctx_event(ctx_id, pc, dim=None):
    del dim
    instr = build_set_riscv_pc(ctx_id, pc)
    return {
        "type": "instruction",
        "op": "SET_RISCV_PC",
        "execute": "context",
        "encoded": _hex_instruction(instr),
        "fields": {
            "cmd": SET_RISCV_PC,
            "pc_id": ctx_id,
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
        },
    }


def sb_write_event(case, addr, lanes_bf16, source):
    assert len(lanes_bf16) == NUM_LANES
    return {
        "type": "sb_write",
        "case": case,
        "addr": addr,
        "lanes_bf16": [value.lower() for value in lanes_bf16],
        "source": source,
    }


def simulate_event(case, op, writes, source):
    return {
        "type": "simulate",
        "case": case,
        "op": op,
        "execute": "simulated",
        "reason": "unsupported_pim_instruction",
        "source": source,
        "writes": writes,
    }


def check_event(case, addr, kind, expected_lanes_bf16, max_ulp=0, source=None):
    event = {
        "type": "check",
        "case": case,
        "target": "SB",
        "addr": addr,
        "kind": kind,
        "expected_lanes_bf16": [value.lower() for value in expected_lanes_bf16],
        "max_ulp": max_ulp,
    }
    if source:
        event["source"] = source
    return event


def _parse_contract_pc(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def _add_context_preamble(instruction_events):
    chmask_to_ctx = {}
    pc_to_id = {}
    next_chmask_ctx = 0
    next_pc_id = 0

    for event in instruction_events:
        if event.get("type") != "instruction":
            continue
        fields = event.get("fields", {})
        if event.get("execute") == "noop" and "chmask" in fields:
            chmask = int(fields["chmask"], 0) if isinstance(fields["chmask"], str) else int(fields["chmask"])
            if _mask_to_direct_selector(chmask) is None and chmask not in chmask_to_ctx:
                if next_chmask_ctx >= MASK_CTX_ENTRIES:
                    raise ValueError("Too many unique non-direct channel masks for fixed 64-bit ISA mask context table")
                chmask_to_ctx[chmask] = next_chmask_ctx
                next_chmask_ctx += 1
        elif event.get("op") == "RISCV":
            pc = _parse_contract_pc(fields.get("pc", 0))
            if pc not in pc_to_id:
                if next_pc_id >= (1 << ISA_SELECTOR_WIDTH):
                    raise ValueError("Too many unique RISCV PCs for fixed 64-bit ISA PC table")
                pc_to_id[pc] = next_pc_id
                next_pc_id += 1

    preamble = [
        set_chmask_event(ctx_id, chmask)
        for chmask, ctx_id in chmask_to_ctx.items()
    ]
    preamble.extend(
        set_riscv_ctx_event(pc_id, pc)
        for pc, pc_id in pc_to_id.items()
    )

    for event in instruction_events:
        fields = event.get("fields", {})
        if event.get("execute") == "noop" and "chmask" in fields:
            chmask = int(fields["chmask"], 0) if isinstance(fields["chmask"], str) else int(fields["chmask"])
            direct_sel = _mask_to_direct_selector(chmask)
            if direct_sel is None:
                ctx_id = chmask_to_ctx[chmask]
                mask_sel = MASK_SEL_CTX_BASE + ctx_id
                fields["mask_ctx_id"] = ctx_id
            else:
                mask_sel = direct_sel
            fields["mask_sel"] = mask_sel
            event["encoded"] = _hex_instruction(build_raw_instruction(
                int(fields["cmd"]),
                mask_sel=mask_sel,
                opsize=int(fields.get("opsize", 0)),
                bank=int(fields.get("bank", 0)),
                ro=int(fields.get("ro", fields.get("r0", 0))),
                co=int(fields.get("co", 0)),
                r1=int(fields.get("r1", fields.get("sb", 0))),
                flags=int(fields.get("flags", 0)),
            ))
        elif event.get("op") == "RISCV":
            pc = _parse_contract_pc(fields.get("pc", 0))
            dim = int(fields.get("dim", 0))
            pc_id = pc_to_id[pc]
            fields["pc_id"] = pc_id
            event["encoded"] = _hex_instruction(build_instruction(
                RISCV,
                int(fields.get("opsize", 0)),
                int(fields.get("r0", 0)),
                int(fields.get("r1", 0)),
                pc=pc,
                dim=dim,
                pc_id=pc_id,
            ))

    return preamble + instruction_events


def rmsnorm_case(name, source, x_values, layout, include_acc=True):
    opsize = layout["opsize"]
    if not 1 <= opsize <= OPSIZE_MAX:
        raise ValueError(f"RMSNorm contract opsize must be 1..{OPSIZE_MAX}, got {opsize}")

    dim = opsize * NUM_LANES
    x = torch.as_tensor(x_values).flatten()[:dim].to(torch.bfloat16)
    if x.numel() != dim:
        raise ValueError(f"{name}: need {dim} values, got {x.numel()}")

    squared = (x * x).to(torch.bfloat16)
    vectors = pack_chunks(squared, NUM_LANES)
    red_scalar, red_steps = simulate_red_bf16(vectors)
    scale = simulate_rmsnorm_scale_bf16(red_scalar, dim)

    setup = [
        {
            "type": "case",
            "name": name,
            "source": source,
            "phase": "rmsnorm_subgraph",
            "dim": dim,
            "opsize": opsize,
            "riscv_opsize": 1,
            "layout": layout,
            "contract": {
                "unsupported_pim_is_simulated_by_sb_mutation": True,
                "hardware_supported_ops": HARDWARE_SUPPORTED_OPS,
            },
            "golden": {
                "red_scalar_bf16": bf16_hex(red_scalar),
                "riscv_scale_bf16": bf16_hex(scale),
                "red_steps": red_steps,
            },
        },
        {
            "type": "tensor",
            "case": name,
            "name": "x",
            "dtype": "bf16",
            "shape": [dim],
            "preview_lanes_bf16": tensor_to_bf16_hex(x[:min(dim, NUM_LANES)]),
        },
        simulate_event(
            name,
            "PIM_MAC_BK_BK_RD_MAC",
            [
                {
                    "target": "SB",
                    "addr": layout["data_base"] + index,
                    "lanes_bf16": lanes,
                    "description": "x_squared_vector",
                }
                for index, lanes in enumerate(vectors)
            ],
            "simulator_computed_x_squared",
        ),
        sb_write_event(name, layout["red_addr"], zero_lanes(), "clear_red_destination"),
        sb_write_event(name, layout["riscv_addr"], zero_lanes(), "clear_riscv_destination"),
    ]

    instructions = []
    checks = []
    if include_acc:
        acc_src_lanes = vectors[0]
        acc_dst_lanes = zero_lanes()
        setup.extend([
            sb_write_event(name, layout["acc_src"], acc_src_lanes, "acc_smoke_source"),
            sb_write_event(name, layout["acc_dst"], acc_dst_lanes, "acc_smoke_destination"),
        ])
        instructions.append(instruction_event(
            "ACC",
            opsize=1,
            rd=layout["acc_dst"],
            rs=layout["acc_src"],
            case=name,
            note="contract smoke check for hardware-supported ACC",
        ))
        checks.append(check_event(
            name,
            layout["acc_dst"],
            "bf16_lanes",
            acc_src_lanes,
            source="ACC zero+source result",
        ))

    instructions.extend([
        instruction_event("RED", opsize=opsize, rd=layout["red_addr"], rs=layout["data_base"], case=name),
        instruction_event(
            "RISCV",
            opsize=1,
            rd=layout["riscv_addr"],
            rs=layout["red_addr"],
            pc=PC_RMSNORM_SCALE,
            dim=dim,
            case=name,
        ),
    ])
    checks.extend([
        check_event(
            name,
            layout["red_addr"],
            "scalar_lane0_zero_rest",
            [bf16_hex(red_scalar)] + zero_lanes()[1:],
            max_ulp=1,
            source="RED hardware scalar",
        ),
        check_event(
            name,
            layout["riscv_addr"],
            "bf16_broadcast",
            [bf16_hex(scale)] * NUM_LANES,
            max_ulp=1,
            source="RISCV RMSNorm scale",
        ),
    ])
    return setup, instructions, checks


def single_channel_fixture_x():
    try:
        from test_single_channel import get_single_channel_input_x
    except ImportError:
        torch.manual_seed(42)
        return torch.randn((1, 1, 4096)) * 0.1
    return get_single_channel_input_x()


def metadata_event(mode):
    return {
        "type": "metadata",
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "mode": mode,
        "lanes_per_sb_entry": NUM_LANES,
        "instruction_width_bits": REQUEST_WIDTH,
        "riscv_dim_width_bits": RISCV_DIM_WIDTH,
        "selector_width_bits": ISA_SELECTOR_WIDTH,
        "opsize_field_width_bits": ISA_OPSIZE_WIDTH,
        "opsize_datapath_width_bits": OPSIZE_DATAPATH_WIDTH,
        "mask_context_entries": MASK_CTX_ENTRIES,
        "isa_encoding": "fixed64_typed_v3",
        "max_direct_opsize": OPSIZE_MAX,
        "hardware_supported_ops": HARDWARE_SUPPORTED_OPS,
        "unsupported_pim_policy": "live_hardware_conditioned_shared_buffer_mutation",
        "unsupported_pim_policy_note": (
            "JSONL carries simulator SB mutation templates; cocotb must rebuild "
            "later dependent mutations from hardware-produced PNM/RISCV outputs "
            "before writing them into the RTL shared buffer."
        ),
    }


def generate_rmsnorm_contract(trace_file, opsize=4, seed=123, include_random=True, include_simulator=True):
    trace_file = Path(trace_file)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    events = [
        metadata_event("rmsnorm"),
        {
            "type": "program",
            "name": "rmsnorm_subgraph_cosim",
            "description": "Setup mutations are emitted before the hardware instruction program.",
        },
    ]
    all_setup = []
    all_instructions = []
    all_checks = []

    if include_random:
        torch.manual_seed(seed)
        x_random = torch.randn(opsize * NUM_LANES, dtype=torch.bfloat16)
        setup, instructions, checks = rmsnorm_case(
            "rmsnorm_random",
            "random",
            x_random,
            {
                "opsize": opsize,
                "data_base": 0,
                "red_addr": 200,
                "riscv_addr": 201,
                "acc_src": 210,
                "acc_dst": 211,
            },
        )
        all_setup.extend(setup)
        all_instructions.extend(instructions)
        all_checks.extend(checks)

    if include_simulator:
        x_sim = single_channel_fixture_x().flatten()
        setup, instructions, checks = rmsnorm_case(
            "rmsnorm_simulator_tensor",
            "cent_simulator.test_single_channel.get_single_channel_input_x",
            x_sim,
            {
                "opsize": opsize,
                "data_base": 16,
                "red_addr": 202,
                "riscv_addr": 203,
                "acc_src": 212,
                "acc_dst": 213,
            },
        )
        all_setup.extend(setup)
        all_instructions.extend(instructions)
        all_checks.extend(checks)

    if not all_instructions:
        raise ValueError("No RMSNorm cases selected")

    events.extend(all_setup)
    events.extend(_add_context_preamble(all_instructions))
    events.append(instruction_event("ISR_EOC"))
    events.extend(all_checks)

    with trace_file.open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, sort_keys=True) + "\n")
    return trace_file


def _single_channel_args(trace_file):
    class Args:
        pass

    args = Args()
    args.pim_compute = True
    args.op_trace = True
    args.trace_prepare = False
    args.trace_norm = True
    args.trace_fc_kqvo = True
    args.trace_attention = True
    args.trace_softmax = True
    args.trace_fc_ffn = True
    args.trace_activation = True
    args.model = "Llama-7B"
    args.seqlen = 1
    args.FC_devices = 1
    args.embedding = False
    args.only_FC = False
    args.only_trace = False
    args.model_parallel = False
    args.pipeline_parallel = True
    args.channels_per_block = 1
    args.num_channels = 1
    args.GEMV = "no-reuse"
    args.reuse_size = 2
    args.max_seq_len = 1
    args.inter_device_attention = False
    args.DRAM_column = 1024
    args.DRAM_row = 1024 * 16
    args.burst_length = 16
    args.num_banks = 16
    args.threads = 1
    args.trace_file = str(trace_file)
    return args


def _hex_instruction(instr):
    return f"0x{instr:0{INSTR_HEX_WIDTH}x}"


def _bf16_lanes_from_tensor(values):
    return tensor_to_bf16_hex(torch.as_tensor(values).flatten()[:NUM_LANES])


def _sb_write_dict(addr, lanes_bf16, description):
    return {
        "target": "SB",
        "addr": int(addr),
        "lanes_bf16": [value.lower() for value in lanes_bf16],
        "description": description,
    }


def _parse_chmask(token):
    value = int(token, 0)
    return (value >> 16) & 0xFFFF, value & 0xFFFF, value


def _pim_instruction_event(line, line_no):
    parts = line.split()
    if len(parts) < 2 or parts[0] != "AiM":
        return None

    op = parts[1]
    if op == "SYNC":
        instr = build_raw_instruction(PIM_OPCODES["SYNC"])
        return {
            "type": "instruction",
            "op": "ISR_SYNC",
            "execute": "control",
            "encoded": _hex_instruction(instr),
            "legacy": {"line": line_no, "raw": line},
            "fields": {"cmd": PIM_OPCODES["SYNC"], "opsize": 0},
        }
    if op == "EOC":
        instr = build_raw_instruction(PIM_OPCODES["EOC"])
        return {
            "type": "instruction",
            "op": "ISR_EOC",
            "execute": "control",
            "encoded": _hex_instruction(instr),
            "legacy": {"line": line_no, "raw": line},
            "fields": {"cmd": PIM_OPCODES["EOC"], "opsize": 0},
        }

    cmd = PIM_OPCODES.get(op)
    if cmd is None:
        return None

    opsize = 0
    bank = 0
    ro = 0
    co = 0
    ch_high = 0
    ch_low = 0
    chmask = 0

    if op in {"WR_BIAS", "RD_MAC", "RD_AF"} and len(parts) >= 4:
        opsize = int(parts[2], 0)
        ch_high, ch_low, chmask = _parse_chmask(parts[3])
    elif op == "AF" and len(parts) >= 3:
        ch_high, ch_low, chmask = _parse_chmask(parts[2])
    elif op in {"MAC_ABK", "EWMUL"} and len(parts) >= 5:
        opsize = int(parts[2], 0)
        ch_high, ch_low, chmask = _parse_chmask(parts[3])
        ro = int(parts[4], 0)
        if len(parts) >= 6:
            co = int(parts[5], 0)
    elif op in {"COPY_BKGB", "COPY_GBBK"} and len(parts) >= 6:
        opsize = int(parts[2], 0)
        ch_high, ch_low, chmask = _parse_chmask(parts[3])
        bank = int(parts[4], 0)
        ro = int(parts[5], 0)
    elif op == "WR_GB" and len(parts) >= 5:
        opsize = int(parts[2], 0)
        co = int(parts[3], 0)
        ch_high, ch_low, chmask = _parse_chmask(parts[4])
    elif op == "EWADD" and len(parts) >= 5:
        opsize = int(parts[2], 0)
        ro = int(parts[3], 0)
        co = int(parts[4], 0)
    elif op == "WR_ABK" and len(parts) >= 5:
        opsize = int(parts[2], 0)
        ch_high, ch_low, chmask = _parse_chmask(parts[3])
        ro = int(parts[4], 0)
    else:
        return None

    instr = build_raw_instruction(
        cmd,
        chmask_id=0,
        opsize=opsize,
        bank=bank,
        ro=ro,
        co=co,
    )
    return {
        "type": "instruction",
        "op": f"PIM_{op}",
        "execute": "noop",
        "reason": "unsupported_pim_instruction",
        "encoded": _hex_instruction(instr),
        "legacy": {"line": line_no, "raw": line},
        "fields": {
            "cmd": cmd,
            "chmask": f"0x{chmask:08x}",
            "opsize": opsize,
            "bank": bank,
            "ro": ro,
            "co": co,
        },
    }


def _parse_legacy_trace_line(raw_line, line_no):
    line = raw_line.strip()
    if not line:
        return None
    parts = line.split()
    op = parts[0]
    event = {
        "type": "instruction",
        "legacy": {"line": line_no, "raw": line},
    }

    if op == "PNM_RED" and len(parts) == 4:
        opsize = int(parts[1], 0)
        event.update({"op": "RED", "execute": "hardware"})
        rd = int(parts[2], 0)
        rs = int(parts[3], 0)
        event["fields"] = {"cmd": RED, "opsize": opsize, "r0": rd, "r1": rs}
        if opsize <= OPSIZE_MAX:
            event["encoded"] = _hex_instruction(build_instruction(RED, opsize, rd, rs))
        else:
            event["reason"] = "opsize_exceeds_decoder_field"
    elif op == "PNM_ACC" and len(parts) == 5:
        opsize = int(parts[1], 0)
        rd = int(parts[2], 0)
        rs1 = int(parts[3], 0)
        rs2 = int(parts[4], 0)
        event.update({"op": "ACC", "execute": "hardware"})
        event["fields"] = {"cmd": ACC, "opsize": opsize, "r0": rd, "r1": rs1, "rs2": rs2}
        if opsize <= OPSIZE_MAX and rs2 == rd:
            event["encoded"] = _hex_instruction(build_instruction(ACC, opsize, rd, rs1))
        else:
            event["reason"] = "hardware_acc_uses_r0_as_destination_and_second_source"
    elif op == "PNM_RISCV" and len(parts) == 5:
        legacy_opsize = int(parts[1], 0)
        pc = int(parts[2], 0)
        rd = int(parts[3], 0)
        rs = int(parts[4], 0)
        dim = legacy_opsize * NUM_LANES
        riscv_opsize = 1 if pc == PC_RMSNORM_SCALE else legacy_opsize
        event.update({"op": "RISCV", "execute": "hardware"})
        event["fields"] = {
            "cmd": RISCV,
            "legacy_opsize": legacy_opsize,
            "opsize": riscv_opsize,
            "dim": dim,
            "pc": f"0x{pc:08x}",
            "r0": rd,
            "r1": rs,
        }
        opsize = riscv_opsize
        if opsize <= OPSIZE_MAX:
            event["encoded"] = _hex_instruction(build_instruction(RISCV, opsize, rd, rs, pc=pc, dim=dim))
            if pc == PC_RMSNORM_SCALE:
                event["semantic_note"] = "RISCV opsize is an operation count; dim is passed separately"
        else:
            event["reason"] = "opsize_exceeds_decoder_field"
    else:
        pim_event = _pim_instruction_event(line, line_no)
        if pim_event is not None:
            return pim_event
        event.update({"op": op.replace("AiM", "PIM"), "execute": "noop", "reason": "unsupported_instruction"})
    return event


def generate_single_channel_contract(trace_file):
    trace_file = Path(trace_file)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w+", suffix=".legacy.trace", delete=False) as tmp:
        legacy_trace = Path(tmp.name)

    try:
        from Llama import TransformerBlockLlama
        from test_single_channel import get_test_inputs

        dic_model = get_test_inputs()
        args = _single_channel_args(legacy_trace)
        block = TransformerBlockLlama(dic_model, args)
        block.cosim_trace_events = []
        block.memory_mapping()
        block.memory_mapping_verification()
        sa_aim = block.self_attention_aim()
        out_aim = block.FFN_aim(sa_aim)
        if hasattr(block, "file"):
            block.file.flush()
            block.file.close()

        events = [
            metadata_event("single_channel_self_attention_ffn"),
            {
                "type": "program",
                "name": "self_attention_aim_plus_FFN_aim",
                "description": (
                    "Full simulator trace translated into JSONL. Events marked hardware "
                    "are encodable by the current decoder; simulated events require "
                    "shared-buffer or DRAM state mutation before they can be replayed by RTL."
                ),
            },
        ]
        pnm_events = iter(block.cosim_trace_events)
        last_red = None
        red_source_index = 0
        with legacy_trace.open("r", encoding="utf-8") as legacy:
            for line_no, raw_line in enumerate(legacy, start=1):
                event = _parse_legacy_trace_line(raw_line, line_no)
                if not event:
                    continue

                if event.get("op") == "RED":
                    red_source_index += 1
                    red_record = next(pnm_events, None)
                    if red_record is None or red_record.get("kind") != "RED":
                        raise RuntimeError(f"Missing simulator RED record for legacy line {line_no}")

                    writes = [
                        _sb_write_dict(reg, _bf16_lanes_from_tensor(value), "RED source vector from simulator SB")
                        for reg, value in zip(red_record["src_regs"], red_record["src_values"])
                    ]
                    sim_event = simulate_event(
                        "single_channel",
                        "SB_STATE_BEFORE_RED",
                        writes,
                        "cent_simulator.shared_buffer.RED_inputs",
                    )
                    sim_event["red_source_index"] = red_source_index
                    if red_source_index > 1:
                        sim_event["requires_live_hardware_state"] = True
                        sim_event["live_dependency"] = "previous RISCV RMSNorm output"
                    events.append(sim_event)

                    event["fields"].update({
                        "opsize": int(red_record["opsize"]),
                        "r0": int(red_record["rd"]),
                        "r1": int(red_record["rs"]),
                    })
                    event["encoded"] = _hex_instruction(build_instruction(
                        RED,
                        int(red_record["opsize"]),
                        int(red_record["rd"]),
                        int(red_record["rs"]),
                    ))
                    event["simulator"] = {
                        "source_register_count": len(red_record["src_regs"]),
                        "source_registers": [int(reg) for reg in red_record["src_regs"]],
                    }
                    events.append(event)

                    expected = _bf16_lanes_from_tensor(red_record["result"])
                    red_check = check_event(
                        "single_channel",
                        int(red_record["rd"]),
                        "scalar_lane0_zero_rest",
                        expected,
                        max_ulp=1,
                        source="RED hardware output",
                    )
                    red_check["red_source_index"] = red_source_index
                    events.append(red_check)
                    last_red = {
                        "rd": int(red_record["rd"]),
                        "expected": expected,
                        "red_source_index": red_source_index,
                    }
                    continue

                if event.get("op") == "RISCV":
                    riscv_record = next(pnm_events, None)
                    if riscv_record is None or riscv_record.get("kind") != "RISCV":
                        raise RuntimeError(f"Missing simulator RISCV record for legacy line {line_no}")

                    dim = int(riscv_record["dim"])
                    riscv_opsize = int(riscv_record["opsize"])
                    rd = int(riscv_record["rd"])
                    rs = int(riscv_record["rs"])
                    pc = int(riscv_record["pc"])
                    event["fields"].update({
                        "legacy_opsize": int(riscv_record["opsize"]),
                        "opsize": riscv_opsize,
                        "dim": dim,
                        "pc": f"0x{pc:08x}",
                        "r0": rd,
                        "r1": rs,
                    })
                    event["encoded"] = _hex_instruction(build_instruction(
                        RISCV,
                        riscv_opsize,
                        rd,
                        rs,
                        pc=pc,
                        dim=dim,
                    ))

                    input_lanes = _bf16_lanes_from_tensor(riscv_record["input"])
                    if last_red is not None and last_red["expected"][0] == input_lanes[0]:
                        if rs != last_red["rd"]:
                            raise RuntimeError(
                                "Simulator RED/RISCV handoff used different SB addresses: "
                                f"RED rd={last_red['rd']} RISCV rs={rs} at legacy line {line_no}"
                            )
                        event["handoff"] = "riscv_reads_previous_red_destination"
                    else:
                        events.append(simulate_event(
                            "single_channel",
                            "SB_STATE_BEFORE_RISCV",
                            [_sb_write_dict(rs, input_lanes, "RISCV source scalar from simulator SB")],
                            "cent_simulator.shared_buffer.RISCV_input",
                        ))

                    event["simulator"] = {
                        "dim": dim,
                        "input_lanes_bf16": input_lanes,
                        "expected_lanes_bf16": _bf16_lanes_from_tensor(riscv_record["result"]),
                    }
                    if last_red is not None:
                        event["rmsnorm_index"] = last_red["red_source_index"]
                    events.append(event)
                    riscv_check = check_event(
                        "single_channel",
                        rd,
                        "bf16_broadcast",
                        event["simulator"]["expected_lanes_bf16"],
                        max_ulp=1,
                        source="RISCV RMSNorm scale using explicit dim",
                    )
                    if last_red is not None:
                        riscv_check["rmsnorm_index"] = last_red["red_source_index"]
                    events.append(riscv_check)
                    last_red = None
                    continue

                events.append(event)
        body_instruction_events = [
            event for event in events[2:] if event.get("type") == "instruction"
        ]
        contextualized = _add_context_preamble(body_instruction_events)
        context_preamble = contextualized[:len(contextualized) - len(body_instruction_events)]
        events = events[:2] + context_preamble + events[2:]
        events.append(instruction_event("ISR_EOC"))

        golden_out = dic_model["out"]
        sim_out = out_aim
        diff = (torch.as_tensor(sim_out).to(torch.float32) - torch.as_tensor(golden_out).to(torch.float32)).abs()
        events.append({
            "type": "final_tensor_check",
            "case": "single_channel",
            "name": "out",
            "source": "test_single_channel.py golden out",
            "shape": list(golden_out.shape),
            "dtype": "bf16",
            "simulated_bf16": tensor_to_bf16_hex(sim_out),
            "golden_bf16": tensor_to_bf16_hex(golden_out),
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "atol": 0.05,
        })

        instruction_events = [event for event in events if event.get("type") == "instruction"]
        events[0]["instruction_count"] = len(instruction_events)
        events[0]["hardware_instruction_count"] = sum(
            1 for event in instruction_events if event.get("execute") == "hardware"
        )
        events[0]["noop_instruction_count"] = sum(
            1 for event in instruction_events if event.get("execute") == "noop"
        )
        events[0]["context_instruction_count"] = sum(
            1 for event in instruction_events if event.get("execute") == "context"
        )
        events[0]["mutation_event_count"] = sum(
            1 for event in events if event.get("type") in {"simulate", "sb_write"}
        )

        with trace_file.open("w", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")
    finally:
        try:
            os.unlink(legacy_trace)
        except OSError:
            pass

    return trace_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, help="Output JSONL trace path")
    parser.add_argument(
        "--mode",
        choices=["rmsnorm", "single-channel"],
        default="rmsnorm",
        help="Trace contract to generate",
    )
    parser.add_argument("--opsize", type=int, default=4, help="RMSNorm subgraph vector count")
    parser.add_argument("--seed", type=int, default=123, help="Random RMSNorm case seed")
    parser.add_argument("--no-random", action="store_true", help="Omit random RMSNorm case")
    parser.add_argument("--no-simulator", action="store_true", help="Omit simulator tensor RMSNorm case")
    args = parser.parse_args()

    if args.mode == "rmsnorm":
        path = generate_rmsnorm_contract(
            args.trace_file,
            opsize=args.opsize,
            seed=args.seed,
            include_random=not args.no_random,
            include_simulator=not args.no_simulator,
        )
    else:
        path = generate_single_channel_contract(args.trace_file)
    print(path)


if __name__ == "__main__":
    main()
