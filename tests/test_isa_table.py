"""The ISA table (npu_model/isa/atlas_isa.toml) is well formed, and the instruction classes encode
the words of their table entries."""
import pytest

import npu_model.configs.isa_definition  # noqa: F401  (registers the instructions)
from npu_model.isa import IsaSpec
from npu_model.isa.table import PATH, TableError, load
from tests.helpers import words

TABLE = load()


def test_classes_and_table_name_the_same_mnemonics() -> None:
    assert sorted(IsaSpec.operations) == sorted(TABLE.entries)


@pytest.mark.parametrize("mnemonic", sorted(IsaSpec.operations))
def test_instruction_matches_its_entry_only(mnemonic: str) -> None:
    for word in words(IsaSpec.operations[mnemonic]):
        hits = [e.mnemonic for e in TABLE.entries.values() if word & e.mask == e.match]
        assert hits == [mnemonic], f"{mnemonic}: {word:#010x} matches {hits}"
        assert TABLE.decode(word) is TABLE.entries[mnemonic]


def test_kinds_fix_the_operand_bits_they_do_not_use() -> None:
    vadd, vpack, matmul = (TABLE.entries[m] for m in ("vadd.bf16", "vpack.bf16.fp8", "vmatmul.mxu1"))
    assert TABLE.decode(vadd.match | 2 << 7) is vadd
    assert TABLE.decode(vadd.match | 1 << 7) is None         # odd BF16 tile register
    assert TABLE.decode(vpack.match | 31 << 13) is vpack
    assert TABLE.decode(vpack.match | 32 << 13) is None      # e32
    assert TABLE.decode(matmul.match | 1 << 7) is matmul
    assert TABLE.decode(matmul.match | 2 << 7) is None       # acc2
    seli = TABLE.entries["seli"]
    assert TABLE.decode(seli.match | 255 << 16) is seli
    assert TABLE.decode(seli.match | 256 << 16) is None      # uimm8


@pytest.mark.parametrize("mnemonic, old, new, error", [
    ("sub", '"sub', '"add', "add: two entries"),
    ("sub", "funct7 = 0x20", "funct7 = 0x00", "add and sub match the same words"),
    ("sub", "funct7 = 0x20", "funct7 = 0x20, ch = 0", "sub: field ch overlaps another field"),
    ("sub", "funct7 = 0x20 }", "funct7 = 0x20 }\nfixed = { funct7 = 0x20 }", "sub: a field is used twice"),
    ("sub", "funct7 = 0x20", "funct7 = 0x80", "sub: funct7 = 128"),
    ("sub", 'rs2 = "x" }', 'shamt = "x" }', "sub: no field shamt in format R"),
    ("sub", 'rs2 = "x" }', 'rs2 = "z" }', "sub: operand kinds"),
    ("sub", "x(rs1), x(rs2)", "x(rs1)", "sub: syntax operands"),
    ("sub", "x(rs1), x(rs2)", "x(rs1), m(rs2)", "sub: syntax register prefixes"),
    ("sub", 'unit = "scalar"', 'unit = "fpu"', "sub: unit fpu"),
    ("sub", "opcode = 0x33", "opcode = 0x57", "sub: opcode: not RV32I or custom"),
    ("vmov", "fixed = { vs2 = 0 }\n", "", "vmov: a bit of a custom opcode is in no field"),
    ("seld", 'rd = "e"', 'rd = "m"', "seld: rd: m does not fit"),
])
def test_malformed_tables_fail(tmp_path, mnemonic: str, old: str, new: str, error: str) -> None:
    text = PATH.read_text()
    start = text.index(f'mnemonic = "{mnemonic}"')
    end = text.index("[[instruction]]", start)
    assert old in text[start:end]
    path = tmp_path / "isa.toml"
    path.write_text(text[:start] + text[start:end].replace(old, new) + text[end:])
    with pytest.raises(TableError, match=error):
        load(path)
