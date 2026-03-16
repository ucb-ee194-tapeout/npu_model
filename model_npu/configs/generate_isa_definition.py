from model_npu.control.isa_parse import instr as instr_parse
from model_npu.isa import instr as instr_sim


def generate_isa_def(decorator: str):
    if decorator == "parse":
        return instr_parse
    elif decorator == "sim":
        return instr_sim
    else:
        return ValueError("invalid decorator")
