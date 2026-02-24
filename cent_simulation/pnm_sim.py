# cent_simulation/pnm_sim.py

class PNM:
    """
    PNM Class containing Orchestrator unit instructions (Reduction, RISC-V, etc.)
    Designed to be inherited by the TransformerBlockLlama class along with PIM class
    """
    
    # Define timing constants for your new units if you want to track latency
    # You might want to initialize this in a separate init or just append to self.timing_constant later
    
    def PNM_RISCV_only_trace(self, instruction_id, cycles):
        """
        Simulate a RISC-V instruction on the PNM unit.
        """
        # Update internal stats (assuming self.time was init by PIM class)
        # You might need to add "PNM_RISCV" to self.time keys in PIM.__init__ or handle it safely here
        if "PNM_RISCV" in self.time:
            self.time["PNM_RISCV"] += cycles

        # Write to the trace file
        # Format: PNM_RISCV <ID> <CYCLES>
        self.file.write(f"PNM_RISCV {instruction_id} {cycles}\n")

    def PNM_RED_only_trace(self, channel_mask, src_row, dst_row, size):
        """
        Simulate a Reduction operation (e.g., Sum) in the PNM unit.
        """
        # Update stats
        if "PNM_RED" in self.time:
             self.time["PNM_RED"] += 1 # or some latency calculation

        # Write to trace file
        # Format: PNM_RED <MASK> <SRC> <DST> <SIZE>
        self.file.write(f"PNM_RED {channel_mask} {src_row} {dst_row} {size}\n")

class SharedBuffer:
    def __init__(self, num_registers=256):
        self.num_registers = num_registers
        # Model 256-bit registers (each holding 16 elements)
        self.registers = {i: torch.zeros(16, dtype=torch.bfloat16) for i in range(num_registers)}
        self.free_regs = list(range(num_registers))

    def allocate(self, count=1):
        if len(self.free_regs) < count:
            raise RuntimeError(f"Out of Shared Buffer registers. Requested {count}, available {len(self.free_regs)}")
        allocated = self.free_regs[:count]
        self.free_regs = self.free_regs[count:]
        return allocated

    def free(self, regs):
        if isinstance(regs, int):
            regs = [regs]
        self.free_regs.extend(regs)
        # Clear data upon free
        for r in regs:
            self.registers[r] = torch.zeros(16, dtype=torch.bfloat16)