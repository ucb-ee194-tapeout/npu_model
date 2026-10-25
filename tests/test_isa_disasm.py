"""The disassembler gives assembly that assembles to the same words."""
import io
import random
from pathlib import Path

import pytest

import npu_model.configs.programs as programs
from npu_model.isa import TABLE, IsaSpec
from npu_model.isa.disasm import disassemble
from npu_model.isa.table import place
from npu_model.util.converter import input_to_program
from tests.helpers import words

ASM = sorted((Path(__file__).resolve().parents[1] / "npu_model/configs/programs/asm").glob("*.S"))


def assemble(source: str) -> list[int]:
    return [w & 0xFFFFFFFF for w in input_to_program(io.StringIO(source)).assemble()]


@pytest.mark.parametrize("mnemonic", sorted(IsaSpec.operations))
def test_every_instruction_round_trips(mnemonic: str) -> None:
    for word in words(IsaSpec.operations[mnemonic]):
        assert assemble(disassemble([word])) == [word], disassemble([word])


@pytest.mark.parametrize("mnemonic", sorted(TABLE.entries))
def test_words_with_random_operands_round_trip(mnemonic: str) -> None:
    entry, rng = TABLE.entries[mnemonic], random.Random(mnemonic)
    fields = TABLE.formats[entry.format]
    operand_bits = sum(place(fields[f], -1) for f in entry.operands) & ~entry.mask
    for _ in range(64):
        word = entry.match | rng.getrandbits(32) & operand_bits
        assert assemble(disassemble([word])) == [word], disassemble([word])


@pytest.mark.parametrize("path", ASM, ids=lambda p: p.stem)
def test_assembly_programs_round_trip(path: Path) -> None:
    program = assemble(path.read_text())
    assert assemble(disassemble(program)) == program


@pytest.mark.parametrize("name", sorted(programs.__all__))
def test_python_programs_round_trip(name: str) -> None:
    program = [w & 0xFFFFFFFF for w in getattr(programs, name)().assemble()]
    assert assemble(disassemble(program)) == program


def test_branch_targets_get_labels() -> None:
    text = disassemble(assemble("top:\n addi x1, x1, -1\n bne x1, x0, top\n jal x0, end\n nop\nend:\n"))
    assert text.splitlines() == [
        "L0:",
        "    addi x1, x1, -1                          # 0: fff08093",
        "    bne x1, x0, L0                           # 4: fe009ee3",
        "    jal x0, L16                              # 8: 0080006f",
        "    addi x0, x0, 0                           # 12: 00000013",
        "L16:",
    ]


def test_an_illegal_word_is_shown_as_data_and_assembles_again() -> None:
    program = [0x13, 0xFFFFFFFF, 0x0000_0057]
    text = disassemble(program)
    assert text.splitlines()[1] == "    .word 0xffffffff                         # 4: ffffffff, illegal"
    assert assemble(text) == program


@pytest.mark.parametrize("word, error", [("0x00000013", "addi instruction"), ("0x100000000", "32-bit"),
                                         ("-1", "32-bit")])
def test_word_takes_only_a_32_bit_word_that_is_no_instruction(word: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        assemble(f".word {word}\n")
