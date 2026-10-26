"""The VS Code grammar and the LSP linter know every instruction of the ISA table, the pseudo-instructions
and `.word`."""
import json
import re
from pathlib import Path

import pytest

from npu_model.isa import TABLE
from npu_model.lsp.linter import lint_text

GRAMMAR = Path(__file__).resolve().parents[1] / "npu_model/lsp/vscode-extension/syntaxes/npu-asm.tmLanguage.json"
MNEMONICS = [re.compile(p["match"]) for p in json.loads(GRAMMAR.read_text())["repository"]["mnemonic"]["patterns"]]


def highlighted(mnemonic: str) -> bool:
    """A grammar pattern matches the whole mnemonic at the start of a line."""
    return any((hit := p.search(f"    {mnemonic} x1")) is not None and hit.group(0) == mnemonic for p in MNEMONICS)


@pytest.mark.parametrize("mnemonic", [*TABLE.entries, "nop", "li", ".word"])
def test_the_grammar_highlights_every_mnemonic(mnemonic: str) -> None:
    assert highlighted(mnemonic)


@pytest.mark.parametrize("mnemonic", ["vmatmul.mxu0", "vmatpush.acc.fp8.nv", "vadd.bf16.vy", "word"])
def test_the_grammar_does_not_highlight_other_words(mnemonic: str) -> None:
    assert not highlighted(mnemonic)


@pytest.mark.parametrize("source, error", [
    (".word 0xffffffff", None),
    (".word", "expects 1 operand"),
    (".word zz", "Expected integer"),
    (".word 0x100000000", "not a 32-bit word"),
    (".word 0x00000013", "is a addi instruction"),
])
def test_the_linter_checks_word(source: str, error: str | None) -> None:
    messages = [d.message for d in lint_text(source)]
    assert messages == [] if error is None else any(error in m for m in messages), messages
