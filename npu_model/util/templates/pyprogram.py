from npu_model.software.instruction import Instruction
from torch._tensor_str import printoptions # type: ignore — this is annoyingly not publicly exported
import torch

def torch_repr(tensor: torch.Tensor):
    with printoptions(threshold=torch.inf, sci_mode=False, precision=8):
        return 'torch.' + repr(tensor)

def format_python_file(
        program_name: str, 
        file_name: str, 
        timeout: int | None, 
        instructions: list[Instruction], 
        mem_regions: list[tuple[int, torch.Tensor]], 
        golden_result: list[tuple[int, torch.Tensor]]
    ):
    return f"""import torch
from npu_model.software.instruction import Instruction, x, e, m, w, acc
from npu_model.software.program import Program
from npu_model.configs.isa_definition import *

class {program_name}(Program):
    \"\"\"
    Automatically generated from {file_name} by assembler.py 
    \"\"\"
    timeout: int | None = {timeout}

    instructions: list[Instruction] = [
{'\n'.join([f'        {insn.serialize()},' for insn in instructions])}
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [{'\n    ' if len(mem_regions) != 0 else ''}{''.join(f'    ({addr}, {torch_repr(tensor)}),\n    ' for addr,tensor in mem_regions)}]

    golden_result: list[tuple[int, torch.Tensor]] = [{'\n    ' if len(golden_result) != 0 else ''}{''.join(f'    ({addr}, {torch_repr(tensor)}),\n    ' for addr,tensor in golden_result)}]
"""