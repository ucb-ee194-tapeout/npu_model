"""The Atlas ISA table (atlas_isa.toml): load it, check it and give each entry its match and mask.

A word w is the instruction of entry e if ``w & e.mask == e.match``.
"""
import itertools
import re
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

PATH = Path(__file__).with_name("atlas_isa.toml")

# Operand kinds: the register prefix of a register kind, or None.
KINDS = {"x": "x", "e": "e", "m": "m", "mp": "m", "mq": "m", "acc": "acc", "w": "w", "ch": None,
         "imm": None, "uimm": None, "hex": None, "label": None}
UNITS = ("scalar", "lsu", "vpu", "xlu", "mxu1", "nv", "dma")
ROWS = ("rows", "all")
KEYS = {"mnemonic", "format", "opcode", "select", "fixed", "operands", "syntax", "unit", "configs", "m",
        "operation", "ver"}
OPTIONAL = {"select", "fixed", "configs", "m"}
PLACEHOLDER = re.compile(r"\b(?:(x|e|m|acc|w)\((\w+)\)|(\w+))")

type Pieces = tuple[tuple[int, int, int], ...]
"""Bit pieces (hi, lo, at) of a field: word bits hi..lo hold the field's bits from `at` up."""


class TableError(ValueError):
    pass


def place(pieces: Pieces, value: int) -> int:
    """The word bits of a field with the given value."""
    return sum(((value >> at) & ((1 << (hi - lo + 1)) - 1)) << lo for hi, lo, at in pieces)


def take(pieces: Pieces, word: int) -> int:
    """The value of a field in a word."""
    return sum(((word >> lo) & ((1 << (hi - lo + 1)) - 1)) << at for hi, lo, at in pieces)


def width(pieces: Pieces) -> int:
    return max(at + hi - lo + 1 for hi, lo, at in pieces)


@dataclass(frozen=True)
class Entry:
    mnemonic: str
    format: str
    opcode: int
    select: dict[str, int]
    fixed: dict[str, int]
    operands: dict[str, str]
    syntax: str
    unit: str
    configs: tuple[str, ...]
    m: str | None
    operation: str
    ver: tuple[int, int]
    match: int
    mask: int

    @property
    def name(self) -> str:
        """The name of the entry's BitPat in Instructions.scala."""
        return self.mnemonic.upper().replace(".", "_")


@dataclass(frozen=True)
class Table:
    pc: str
    configurations: tuple[str, ...]
    formats: dict[str, dict[str, Pieces]]
    entries: dict[str, Entry]

    @property
    def version(self) -> tuple[int, int]:
        """The ISA version: the highest `ver` of the entries."""
        return max(e.ver for e in self.entries.values())

    @cached_property
    def _by_opcode(self) -> dict[int, list[Entry]]:
        out: dict[int, list[Entry]] = {}
        for e in self.entries.values():
            out.setdefault(e.opcode, []).append(e)
        return out

    def decode(self, word: int) -> Entry | None:
        """The entry of a word, or None if the word is illegal."""
        return next((e for e in self._by_opcode.get(word & 0x7F, []) if word & e.mask == e.match), None)


def _pieces(spec: list) -> Pieces:
    if all(isinstance(v, int) for v in spec):
        spec = [[*spec, 0]]
    return tuple((hi, lo, at) for hi, lo, at in spec)


def _entry(raw: dict, formats: dict[str, dict[str, Pieces]], configurations: tuple[str, ...]) -> Entry:
    mn = raw.get("mnemonic", "?")

    def check(ok: bool, what: str) -> None:
        if not ok:
            raise TableError(f"{mn}: {what}")

    check(KEYS - OPTIONAL <= raw.keys() <= KEYS, f"keys {sorted(raw)}")
    check(raw["format"] in formats, f"format {raw['format']}")
    fields = formats[raw["format"]]
    check(raw["opcode"] & 3 == 3 and raw["opcode"] < 128, "opcode")
    select, fixed, operands = raw.get("select", {}), raw.get("fixed", {}), raw["operands"]
    names = ["opcode", *select, *fixed, *operands]
    check(len(names) == len(set(names)), "a field is used twice")
    used = {"opcode": raw["opcode"]} | select | fixed
    bits = 0
    for field in names:
        check(field in fields, f"no field {field} in format {raw['format']}")
        field_bits = place(fields[field], -1)
        check(not bits & field_bits, f"field {field} overlaps another field")
        bits |= field_bits
    for field, value in used.items():
        check(0 <= value < 1 << width(fields[field]), f"{field} = {value}")
    check(all(kind in KINDS for kind in operands.values()), f"operand kinds {operands}")

    syntax = raw["syntax"]
    check(syntax.split()[0] == mn, "syntax does not start with the mnemonic")
    named = {}
    for prefix, reg_field, field in PLACEHOLDER.findall(syntax.removeprefix(mn)):
        named[reg_field or field] = prefix or None
    check(named.keys() == operands.keys(), f"syntax operands {sorted(named)}")
    check(all(KINDS[operands[f]] == p for f, p in named.items()), "syntax register prefixes")

    check(raw["unit"] in UNITS, f"unit {raw['unit']}")
    configs = tuple(raw.get("configs", configurations))
    check(set(configs) <= set(configurations), f"configs {configs}")
    check(raw.get("m", "rows") in ROWS, f"m {raw.get('m')}")
    check(re.fullmatch(r"\d+\.\d+", raw["ver"]) is not None, f"ver {raw['ver']}")
    major, minor = map(int, raw["ver"].split("."))
    match = sum(place(fields[f], v) for f, v in used.items())
    mask = sum(place(fields[f], -1) for f in used)
    return Entry(mn, raw["format"], raw["opcode"], select, fixed, operands, syntax, raw["unit"], configs,
                 raw.get("m"), raw["operation"], (major, minor), match, mask)


def load(path: Path = PATH) -> Table:
    """Load the table. Raises TableError if an entry is malformed or two entries match one word."""
    raw = tomllib.loads(path.read_text())
    if raw.get("pc") not in ("word", "byte"):
        raise TableError(f"pc {raw.get('pc')}")
    configurations = tuple(raw["configurations"])
    formats = {name: {field: _pieces(spec) for field, spec in fields.items()}
               for name, fields in raw["format"].items()}
    entries: dict[str, Entry] = {}
    for item in raw["instruction"]:
        entry = _entry(item, formats, configurations)
        if entry.mnemonic in entries:
            raise TableError(f"{entry.mnemonic}: two entries")
        entries[entry.mnemonic] = entry
    for a, b in itertools.combinations(entries.values(), 2):
        if not (a.match ^ b.match) & a.mask & b.mask:
            raise TableError(f"{a.mnemonic} and {b.mnemonic} match the same words")
    return Table(raw["pc"], configurations, formats, entries)
