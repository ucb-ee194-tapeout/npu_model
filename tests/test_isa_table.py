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


@pytest.mark.parametrize("old, new, error", [
    ('"sub', '"add', "add: two entries"),
    ("funct7 = 0x20", "funct7 = 0x00", "add and sub match the same words"),
    ("funct7 = 0x20", "funct7 = 0x20, ch = 0", "sub: field ch overlaps another field"),
    ("funct7 = 0x20 }", "funct7 = 0x20 }\nfixed = { funct7 = 0x20 }", "sub: a field is used twice"),
    ("funct7 = 0x20", "funct7 = 0x80", "sub: funct7 = 128"),
    ('rs2 = "x" }', 'shamt = "x" }', "sub: no field shamt in format R"),
    ('rs2 = "x" }', 'rs2 = "z" }', "sub: operand kinds"),
    ("x(rs1), x(rs2)", "x(rs1)", "sub: syntax operands"),
    ("x(rs1), x(rs2)", "x(rs1), m(rs2)", "sub: syntax register prefixes"),
    ('unit = "scalar"', 'unit = "fpu"', "sub: unit fpu"),
])
def test_malformed_tables_fail(tmp_path, old: str, new: str, error: str) -> None:
    text = PATH.read_text()
    start = text.index('mnemonic = "sub"')
    end = text.index("[[instruction]]", start)
    assert old in text[start:end]
    path = tmp_path / "isa.toml"
    path.write_text(text[:start] + text[start:end].replace(old, new) + text[end:])
    with pytest.raises(TableError, match=error):
        load(path)
