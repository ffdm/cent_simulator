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
OPSIZE_MAX = 63
SCHEMA = "cent-cosim-jsonl"
SCHEMA_VERSION = 1

RISCV = 16
ISR_EOC = 18
RED = 21
ACC = 22

OPCODES = {
    "RISCV": RISCV,
    "ISR_EOC": ISR_EOC,
    "RED": RED,
    "ACC": ACC,
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


def simulate_rmsnorm_scale_bf16(red_scalar_bf16, opsize):
    sum_f32 = np.float32(bf16_int_to_float(red_scalar_bf16))
    dim_f32 = np.float32(opsize * NUM_LANES)
    eps_f32 = np.float32(1e-5)
    one_f32 = np.float32(1.0)
    mean_f32 = sum_f32 / dim_f32
    inner_f32 = mean_f32 + eps_f32
    root_f32 = np.sqrt(inner_f32).astype(np.float32)
    scale_f32 = one_f32 / root_f32
    return float_to_bf16_int(scale_f32)


def build_instruction(cmd, opsize=0, r0=0, r1=0, pc=0):
    if not 0 <= opsize <= OPSIZE_MAX:
        raise ValueError(f"opsize must fit in 6 bits: {opsize}")
    instr = 0
    instr |= (cmd & 0x1F) << 77
    instr |= ((pc >> 16) & 0xFFFF) << 61
    instr |= (pc & 0xFFFF) << 45
    instr |= (opsize & 0x3F) << 39
    instr |= (r0 & 0x7FF) << 21
    instr |= (r1 & 0x7FF)
    return instr


def instruction_event(op, opsize=0, rd=0, rs=0, pc=0, case=None, note=None):
    cmd = OPCODES[op]
    instr = build_instruction(cmd, opsize=opsize, r0=rd, r1=rs, pc=pc)
    event = {
        "type": "instruction",
        "op": op,
        "execute": "control" if op == "ISR_EOC" else "hardware",
        "encoded": f"0x{instr:021x}",
        "fields": {
            "cmd": cmd,
            "opsize": opsize,
            "r0": rd,
            "r1": rs,
            "pc": f"0x{pc:08x}",
        },
    }
    if case is not None:
        event["case"] = case
    if op == "ACC":
        event["semantics"] = "SB[r0 + i] = SB[r0 + i] + SB[r1 + i]"
    if note:
        event["note"] = note
    return event


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
    scale = simulate_rmsnorm_scale_bf16(red_scalar, opsize)

    setup = [
        {
            "type": "case",
            "name": name,
            "source": source,
            "phase": "rmsnorm_subgraph",
            "dim": dim,
            "opsize": opsize,
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
            opsize=opsize,
            rd=layout["riscv_addr"],
            rs=layout["red_addr"],
            pc=PC_RMSNORM_SCALE,
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
        "instruction_width_bits": 82,
        "max_direct_opsize": OPSIZE_MAX,
        "hardware_supported_ops": HARDWARE_SUPPORTED_OPS,
        "unsupported_pim_policy": "mutate_shared_buffer_from_simulator_event",
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
    events.extend(all_instructions)
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


def _parse_legacy_trace_line(raw_line, line_no):
    line = raw_line.strip()
    if not line:
        return None
    parts = line.split()
    op = parts[0]
    event = {
        "type": "legacy_trace_event",
        "line": line_no,
        "raw": line,
        "execute": "simulated",
    }

    if op == "PNM_RED" and len(parts) == 4:
        opsize = int(parts[1], 0)
        event.update({"op": "RED", "opsize": opsize, "rd": int(parts[2], 0), "rs": int(parts[3], 0)})
        if opsize <= OPSIZE_MAX:
            event["execute"] = "hardware"
            event["encoded"] = f"0x{build_instruction(RED, opsize, event['rd'], event['rs']):021x}"
        else:
            event["reason"] = "opsize_exceeds_6_bit_decoder_field"
    elif op == "PNM_ACC" and len(parts) == 5:
        opsize = int(parts[1], 0)
        rd = int(parts[2], 0)
        rs1 = int(parts[3], 0)
        rs2 = int(parts[4], 0)
        event.update({"op": "ACC", "opsize": opsize, "rd": rd, "rs1": rs1, "rs2": rs2})
        if opsize <= OPSIZE_MAX and rs2 == rd:
            event["execute"] = "hardware"
            event["encoded"] = f"0x{build_instruction(ACC, opsize, rd, rs1):021x}"
        else:
            event["reason"] = "hardware_acc_uses_r0_as_destination_and_second_source"
    elif op == "PNM_RISCV" and len(parts) == 5:
        opsize = int(parts[1], 0)
        pc = int(parts[2], 0)
        rd = int(parts[3], 0)
        rs = int(parts[4], 0)
        event.update({"op": "RISCV", "opsize": opsize, "pc": f"0x{pc:08x}", "rd": rd, "rs": rs})
        if opsize <= OPSIZE_MAX:
            event["execute"] = "hardware"
            event["encoded"] = f"0x{build_instruction(RISCV, opsize, rd, rs, pc=pc):021x}"
            if pc == PC_RMSNORM_SCALE and opsize == 1:
                event["semantic_note"] = (
                    "legacy simulator RMSNorm RISCV uses the block dim internally; "
                    "hardware firmware interprets dim as opsize * 16"
                )
        else:
            event["reason"] = "opsize_exceeds_6_bit_decoder_field"
    else:
        event["op"] = op.replace("AiM", "PIM")
        event["reason"] = "unsupported_pim_instruction"
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
        block.memory_mapping()
        block.memory_mapping_verification()
        sa_aim = block.self_attention_aim()
        block.FFN_aim(sa_aim)
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
        with legacy_trace.open("r", encoding="utf-8") as legacy:
            for line_no, raw_line in enumerate(legacy, start=1):
                event = _parse_legacy_trace_line(raw_line, line_no)
                if event:
                    events.append(event)
        events.append(instruction_event("ISR_EOC"))

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
