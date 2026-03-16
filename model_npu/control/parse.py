if __name__ == "__main__":
    from model_npu.control.isa_parse import instr, _generate_chisel_table
    from model_npu.configs.isa_definition import define_isa

    define_isa(instr)
    _generate_chisel_table()
