if __name__ == "__main__":
    from npu_model.control.isa_parse import instr, _generate_chisel_table
    from npu_model.configs.isa_definition import define_isa

    define_isa(instr)
    _generate_chisel_table()
