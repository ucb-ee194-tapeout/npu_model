if __name__ == "__main__":
    from model_npu.configs.isa_definition import *
    from model_npu.control.insn.instruction import _generate_chisel_table

    _generate_chisel_table()
