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