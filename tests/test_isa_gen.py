"""The files that npu_model.isa.gen writes hold the encodings of the ISA table, and the green card's
instruction table is up to date."""
import json
import re

from npu_model.isa.gen import CARD, atlas_isa_inc, bitpat, card, instructions_scala, rtl_bitpats
from npu_model.isa.table import load, place

TABLE = load()


def mask_match(bits: str) -> tuple[int, int]:
    return int(bits.replace("0", "1").replace("?", "0"), 2), int(bits.replace("?", "0"), 2)


def test_scala_bitpats_and_version() -> None:
    text = instructions_scala(TABLE)
    pats = dict(re.findall(r'def (\w+) +=\s*BitPat\("b([01?]{32})"\)', text))
    assert {name: mask_match(bits) for name, bits in pats.items()} == \
        {e.name: (e.mask, e.match) for e in TABLE.entries.values()}
    assert f"val ISA_MAJOR = {TABLE.version[0]}\n  val ISA_MINOR = {TABLE.version[1]}\n" in text


def test_card_is_generated() -> None:
    text = CARD.read_text()
    assert text == card(TABLE, text), "python -m npu_model.isa.gen card"
    for e in TABLE.entries.values():
        assert f"| `{e.match:08x}` / `{e.mask:08x}` |" in text, e.mnemonic


def test_atlas_isa_inc() -> None:
    text = atlas_isa_inc(TABLE)
    insns = re.findall(r'^ATLAS_INSN\((\w+), "([\w.]+)", (\w+), 0x([0-9a-f]{8})u, 0x([0-9a-f]{8})u\)$', text, re.M)
    assert [(n, m, f, int(match, 16), int(mask, 16)) for n, m, f, match, mask in insns] == \
        [(e.name, e.mnemonic, e.format, e.match, e.mask) for e in TABLE.entries.values()]
    fields: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for fmt, field, hi, lo, at in re.findall(r"^ATLAS_FIELD\((\w+), (\w+), (\d+), (\d+), (\d+)\)$", text, re.M):
        fields.setdefault((fmt, field), []).append((int(hi), int(lo), int(at)))
    assert fields == {(fmt, f): list(p) for fmt, fs in TABLE.formats.items() for f, p in fs.items()}
    # A B-type offset of -2 sets every offset bit of the word.
    assert place(tuple(fields[("B", "offset")]), -2) == 0xFE000F80


def test_rtl_bitpats_json() -> None:
    data = json.loads(rtl_bitpats(TABLE))
    assert data["isa"] == list(TABLE.version)
    assert data["bitpats"] == {e.name: {"mnemonic": e.mnemonic, "rtl_bitpat": bitpat(e)} for e in TABLE.entries.values()}
