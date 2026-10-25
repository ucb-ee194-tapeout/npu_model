"""Disassembler: Atlas instruction words to assembly that npu_model's assembler reads back.

A branch or jump to an address in the program gets a label, L<address>. Addresses are byte
addresses. A word that is no instruction (illegal) prints as `.word`.
"""
from npu_model.isa import TABLE
from npu_model.isa.table import KINDS, PLACEHOLDER, Entry, Table, take, width


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def _offset(t: Table, e: Entry, word: int) -> int | None:
    """The branch or jump byte offset, or None."""
    field = next((f for f, kind in e.operands.items() if kind == "label"), None)
    if field is None:
        return None
    pieces = t.formats[e.format][field]
    return _signed(take(pieces, word), width(pieces))


def _target(t: Table, e: Entry, address: int, word: int, end: int) -> int | None:
    """The address of a branch or jump target from 0 to the end of the program, or None."""
    offset = _offset(t, e, word)
    if offset is None:
        return None
    target = address + offset
    return target if 0 <= target <= end and target % 4 == 0 else None


def _instruction(t: Table, e: Entry, address: int, word: int, end: int) -> str:
    fields = t.formats[e.format]

    def show(field: str) -> str:
        kind, value = e.operands[field], take(fields[field], word)
        if KINDS[kind].prefix:
            return f"{KINDS[kind].prefix}{value}"
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
    end = 4 * len(words)
    entries = [t.decode(w) for w in words]
    labels = {_target(t, e, 4 * i, w, end) for i, (e, w) in enumerate(zip(entries, words)) if e}
    lines = []
    for i, (e, w) in enumerate(zip(entries, words)):
        if 4 * i in labels:
            lines.append(f"L{4 * i}:")
        text = _instruction(t, e, 4 * i, w, end) if e else f".word {w:#010x}"
        lines.append(f"    {text:<40} # {4 * i}: {w:08x}{'' if e else ', illegal'}")
    if end in labels:
        lines.append(f"L{end}:")
    return "\n".join(lines) + "\n"
