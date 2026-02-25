import torch

# Theoretical RISC-V Firmware PCs
PC_RMSNORM_SCALE = 0x1000
PC_SOFTMAX_SCALE = 0x1004
PC_SOFTMAX_EXP_SUM = 0x1008

class PNM:
    """
    PNM Class containing Orchestrator unit instructions (Reduction, RISC-V, etc.)
    Designed to be inherited by the TransformerBlockLlama class along with PIM class
    """
    
    def PNM_RISCV_only_trace(self, instruction_id, cycles):
        """
        Simulate a RISC-V instruction on the PNM unit.
        """
        if "PNM_RISCV" in self.time:
            self.time["PNM_RISCV"] += cycles
        self.file.write(f"PNM_RISCV {instruction_id} {cycles}\n")

    def PNM_RED_only_trace(self, channel_mask, src_row, dst_row, size):
        """
        Simulate a Reduction operation (e.g., Sum) in the PNM unit.
        """
        if "PNM_RED" in self.time:
             self.time["PNM_RED"] += 1 
        self.file.write(f"PNM_RED {channel_mask} {src_row} {dst_row} {size}\n")

    def RED(self, opsize, rd, rs):
        if not self.only_trace:
            if isinstance(rs, list):
                total_sum = sum(self.shared_buffer.registers[r].sum() for r in rs)
            else:
                total_sum = self.shared_buffer.registers[rs].sum()
            
            rd_reg = rd[0] if isinstance(rd, list) else rd
            # Store in the first element
            self.shared_buffer.registers[rd_reg][0] = total_sum
            
            # TODO: add trace generation

    def ACC(self, opsize, rd, rs):
        if not self.only_trace:
            if isinstance(rs, list) and isinstance(rd, list):
                for i in range(opsize):
                    self.shared_buffer.registers[rd[i]] += self.shared_buffer.registers[rs[i]]
            else:
                self.shared_buffer.registers[rd] += self.shared_buffer.registers[rs]
            
            # TODO: add trace generation

    def EXP(self, opsize, rd, rs):
        if not self.only_trace:
            if isinstance(rs, list) and isinstance(rd, list):
                for i in range(opsize):
                    self.shared_buffer.registers[rd[i]] = torch.exp(self.shared_buffer.registers[rs[i]])
            else:
                self.shared_buffer.registers[rd] = torch.exp(self.shared_buffer.registers[rs])
            
            # TODO: add trace generation

    def RISCV(self, opsize, pc, rd, rs):
        if not self.only_trace:
            rs_reg = rs[0] if isinstance(rs, list) else rs
            rd_reg = rd[0] if isinstance(rd, list) else rd
            
            if pc == PC_RMSNORM_SCALE:
                input_data = self.shared_buffer.registers[rs_reg][0]
                norm = torch.rsqrt(input_data / self.dim + 1e-5)
                # Broadcast the scalar norm to all elements
                self.shared_buffer.registers[rd_reg] = torch.full((16,), norm.item(), dtype=torch.bfloat16)

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
