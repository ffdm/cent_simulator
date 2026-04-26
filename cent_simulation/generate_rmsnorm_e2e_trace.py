#!/usr/bin/env python3
"""Generate the PNM trace consumed by the CENT RMSNorm e2e RTL test."""

import argparse
from pathlib import Path

from pnm_sim import PC_RMSNORM_SCALE, PNM


class TraceOnlyPNM(PNM):
    """Minimal PNM instance that emits op-trace lines without functional state."""

    def __init__(self, trace_file):
        self.only_trace = True
        self.op_trace = True
        self.file = trace_file
        self.time = {}


def generate_trace(trace_file, opsize, rd, rs, riscv_rd=None, riscv_rs=None, riscv_pc=PC_RMSNORM_SCALE):
    trace_file = Path(trace_file)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    if riscv_rd is None:
        riscv_rd = rd + 1
    if riscv_rs is None:
        riscv_rs = rd

    with trace_file.open("w", encoding="utf-8") as f:
        pnm = TraceOnlyPNM(f)
        pnm.RED(opsize=opsize, rd=rd, rs=rs)
        pnm.RISCV(opsize=opsize, pc=riscv_pc, rd=riscv_rd, rs=riscv_rs)
        f.write("ISR_EOC\n")

    return trace_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, help="Output trace path")
    parser.add_argument("--opsize", type=int, default=4)
    parser.add_argument("--rd", type=int, default=200)
    parser.add_argument("--rs", type=int, default=0)
    parser.add_argument("--riscv-rd", type=int, default=None)
    parser.add_argument("--riscv-rs", type=int, default=None)
    parser.add_argument("--riscv-pc", type=lambda value: int(value, 0), default=PC_RMSNORM_SCALE)
    args = parser.parse_args()

    path = generate_trace(
        args.trace_file,
        args.opsize,
        args.rd,
        args.rs,
        riscv_rd=args.riscv_rd,
        riscv_rs=args.riscv_rs,
        riscv_pc=args.riscv_pc,
    )
    print(path)


if __name__ == "__main__":
    main()
