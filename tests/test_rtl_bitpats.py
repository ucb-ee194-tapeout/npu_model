"""The RTL decoder table (src/main/scala/atlas/scalar/Instructions.scala) is the file that
`python -m npu_model.isa.gen scala` writes from the ISA table. Every npu_model instruction encodes to
a word that matches its BitPat and no other BitPat, and every BitPat has one npu_model instruction.
The encoder fields have the widths and positions of ScalarDecoder.scala."""
import itertools
import re
from pathlib import Path

import pytest

import npu_model.configs.isa_definition  # noqa: F401  (registers the instructions)
from npu_model.isa import (TABLE, CSRType, IsaSpec, IType, RType, SBType, SType, UJType, UType, VIType, VLSType,
                           VRType)
from npu_model.isa.gen import instructions_scala
from tests.helpers import words

ATLAS_SRC = Path(__file__).resolve().parents[2] / "src/main/scala/atlas"
INSTRUCTIONS_SCALA = ATLAS_SRC / "scalar/Instructions.scala"

# Word bits (high, low) of each field, as ScalarDecoder.scala and Instructions.scala place them;
# the opcode is [6:0].
FORMATS = {"R": RType, "I": IType, "S": SType, "SB": SBType, "U": UType, "UJ": UJType, "CSR": CSRType,
           "VLS": VLSType, "VR": VRType, "VI": VIType}
FIELDS = {
    "R": {"funct7": [(31, 25)], "rs2": [(24, 20)], "rs1": [(19, 15)], "funct3": [(14, 12)], "rd": [(11, 7)]},
    "I": {"imm": [(31, 20)], "rs1": [(19, 15)], "funct3": [(14, 12)], "rd": [(11, 7)]},
    "S": {"imm": [(31, 25), (11, 7)], "rs2": [(24, 20)], "rs1": [(19, 15)], "funct3": [(14, 12)]},
    "SB": {"imm": [(31, 25), (11, 7)], "rs2": [(24, 20)], "rs1": [(19, 15)], "funct3": [(14, 12)]},
    "U": {"imm": [(31, 12)], "rd": [(11, 7)]},
    "UJ": {"imm": [(31, 12)], "rd": [(11, 7)]},
    "CSR": {"imm": [(31, 20)], "rs1": [(19, 15)], "funct3": [(14, 12)], "rd": [(11, 7)]},
    "VLS": {"imm": [(31, 20)], "rs1": [(19, 15)], "funct2": [(14, 13)], "vd": [(12, 7)]},
    "VR": {"funct7": [(31, 25)], "vs2": [(24, 19)], "vs1": [(18, 13)], "vd": [(12, 7)]},
    "VI": {"imm": [(31, 16)], "funct3": [(15, 13)], "vd": [(12, 7)]},
}


@pytest.fixture(scope="module")
def scala() -> str:
    if not ATLAS_SRC.is_dir():
        pytest.skip("standalone npu_model checkout: no Atlas RTL")
    assert INSTRUCTIONS_SCALA.is_file(), f"{INSTRUCTIONS_SCALA} is missing: update this test"
    return INSTRUCTIONS_SCALA.read_text()


def test_instructions_scala_is_generated(scala: str) -> None:
    assert scala == instructions_scala(TABLE), \
        "Instructions.scala differs from the ISA table: python -m npu_model.isa.gen scala " + str(INSTRUCTIONS_SCALA)


@pytest.fixture(scope="module")
def bitpats(scala: str) -> dict[str, tuple[int, int]]:
    """BitPat name -> (mask, match)."""
    found = re.findall(r'def\s+(\w+)\s*=\s*BitPat\("b([01?_]+)"\)', scala)
    assert len(found) == scala.count('BitPat("'), "a BitPat this test cannot parse"
    pats = {}
    for name, bits in found:
        bits = bits.replace("_", "")
        assert len(bits) == 32 and name not in pats, name
        pats[name] = (int(bits.replace("0", "1").replace("?", "0"), 2), int(bits.replace("?", "0"), 2))
    return pats


@pytest.mark.parametrize("mnemonic", sorted(IsaSpec.operations))
def test_instruction_matches_its_bitpat_only(mnemonic: str, bitpats) -> None:
    name = TABLE.entries[mnemonic].name
    assert name in bitpats, f"{mnemonic}: Instructions.scala has no {name}"
    for word in words(IsaSpec.operations[mnemonic]):
        hits = [n for n, (mask, match) in bitpats.items() if word & mask == match]
        assert hits == [name], f"{mnemonic}: {word:#010x} matches {hits}"


def test_every_bitpat_has_one_instruction(bitpats) -> None:
    assert sorted(TABLE.entries[m].name for m in IsaSpec.operations) == sorted(bitpats)


def test_no_word_matches_two_bitpats(bitpats) -> None:
    for (a, (mask_a, match_a)), (b, (mask_b, match_b)) in itertools.combinations(bitpats.items(), 2):
        assert (match_a ^ match_b) & mask_a & mask_b, f"{a} and {b} match the same words"


def bits(ranges: list[tuple[int, int]]) -> int:
    return sum(((1 << (hi - lo + 1)) - 1) << lo for hi, lo in ranges)


@pytest.mark.parametrize("fmt", FIELDS)
def test_encoder_fields_have_the_decoder_bits(fmt: str) -> None:
    fields = FIELDS[fmt] | {"opcode": [(6, 0)]}
    assert sum(map(bits, fields.values())) == 0xFFFFFFFF  # the fields tile the word
    insn = object.__new__(next(c for c in IsaSpec.operations.values() if issubclass(c, FORMATS[fmt])))
    for field, ranges in fields.items():
        for name in fields:
            setattr(insn, name, 0)
        width = sum(hi - lo + 1 for hi, lo in ranges)
        # B and J immediates are even: their bit 0 is not encoded.
        shift = 1 if fmt in ("SB", "UJ") and field == "imm" else 0
        setattr(insn, field, ((1 << width) - 1) << shift)
        assert insn.to_bytecode() == bits(ranges), f"{fmt} {field}"
