"""Disassembler: Atlas instruction words to assembly that npu_model's assembler reads back.

A branch or jump to an address in the program gets a label, L<address>. Addresses count words or
bytes, as the table's `pc` says.
"""
from npu_model.isa import TABLE
from npu_model.isa.table import KINDS, PLACEHOLDER, Entry, Table, take, width


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def _step(t: Table) -> int:
    """The address step from one instruction to the next."""
    return 1 if t.pc == "word" else 4


def _offset(t: Table, e: Entry, word: int) -> int | None:
    """The branch or jump offset in PC units (a word offset is half the field), or None."""
    field = next((f for f, kind in e.operands.items() if kind == "label"), None)
    if field is None:
        return None
    pieces = t.formats[e.format][field]
    offset = _signed(take(pieces, word), width(pieces))
    return offset // 2 if t.pc == "word" else offset


def _target(t: Table, e: Entry, address: int, word: int, end: int) -> int | None:
    """The address of a branch or jump target from 0 to the end of the program, or None."""
    offset = _offset(t, e, word)
    if offset is None:
        return None
    target = address + offset
    return target if 0 <= target <= end and target % _step(t) == 0 else None


def _instruction(t: Table, e: Entry, address: int, word: int, end: int) -> str:
    fields = t.formats[e.format]

    def show(field: str) -> str:
        kind, value = e.operands[field], take(fields[field], word)
        if KINDS[kind]:
            return f"{KINDS[kind]}{value}"
        if kind == "ch":
            return f"ch{value}"
        if kind == "imm":
            return str(_signed(value, width(fields[field])))
        if kind == "hex":
            return hex(value)
        if kind == "label":
            target = _target(t, e, address, word, end)
            return str(_offset(t, e, word)) if target is None else f"L{target}"
        return str(value)

    return e.mnemonic + PLACEHOLDER.sub(lambda m: show(m.group(2) or m.group(3)), e.syntax.removeprefix(e.mnemonic))


def disassemble(words: list[int], t: Table = TABLE) -> str:
    """The program as assembly, one instruction per line, with its address and word in a comment."""
    step, end = _step(t), len(words) * _step(t)
    entries = [t.decode(w) for w in words]
    labels = {_target(t, e, i * step, w, end) for i, (e, w) in enumerate(zip(entries, words)) if e}
    lines = []
    for i, (e, w) in enumerate(zip(entries, words)):
        if i * step in labels:
            lines.append(f"L{i * step}:")
        text = _instruction(t, e, i * step, w, end) if e else f".word {w:#010x}"
        lines.append(f"    {text:<40} # {i * step}: {w:08x}{'' if e else ', illegal'}")
    if end in labels:
        lines.append(f"L{end}:")
    return "\n".join(lines) + "\n"
