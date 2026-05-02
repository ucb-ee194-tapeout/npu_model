"""
Handles importing JSON from the baremetal test repo.

That is, processes JSON files of the form:
```json
{
    "dram_preloads": [
        {
            "word_offset": int,
            "data": "0x<128 hex chars = 64 bytes = one bf16 word>"
        },
        ...
    ],
    "dram_checks": [
        {
            "word_offset": int,
            "expected": "0x<128 hex chars = 64 bytes = one bf16 word>"
        },
        ...
    ],
    "timeout": int
}
```
into a dataclass with the following members:
 - timeout: int
 - memory_regions: list[tuple(int, torch.Tensor)]
 - golden_result: list[tuple(int, torch.Tensor)]
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

_BASE_ADDR = 0x90000000
_BF16_WORD_BYTES = 32  # one bf16 word = 16 bf16 values = 32 bytes

@dataclass
class ProgramData:
    timeout: int
    memory_regions: list[tuple[int, torch.Tensor]]
    golden_result: list[tuple[int, torch.Tensor]]


def _parse_blob(hex_str: str) -> torch.Tensor:
    h = hex_str.removeprefix("0x")
    n_words = len(h) // 8
    # The blob is a big-endian hex number: word 0 is the rightmost 8 chars (LSB)
    words = [int(h[(n_words - 1 - i) * 8:(n_words - i) * 8], 16) for i in range(n_words)]
    return torch.frombuffer(bytearray(struct.pack(f"<{n_words}I", *words)), dtype=torch.uint8).clone()


def _entries_to_regions(
    entries: list[dict[str, Any]], key: Literal["data", "expected"]
) -> list[tuple[int, torch.Tensor]]:
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: e["word_offset"])

    regions: list[tuple[int, torch.Tensor]] = []
    group_start_word: int = sorted_entries[0]["word_offset"]
    group_chunks: list[torch.Tensor] = []

    for entry in sorted_entries:
        if entry["word_offset"] != group_start_word + len(group_chunks):
            regions.append((_BASE_ADDR + group_start_word * _BF16_WORD_BYTES, torch.cat(group_chunks)))
            group_start_word = entry["word_offset"]
            group_chunks = []
        group_chunks.append(_parse_blob(entry[key]))

    regions.append((_BASE_ADDR + group_start_word * _BF16_WORD_BYTES, torch.cat(group_chunks)))
    return regions


def load_json(source: Path) -> ProgramData:
    with open(source) as f:
        data = json.load(f)

    return ProgramData(
        timeout=int(data["timeout"]),
        memory_regions=_entries_to_regions(data.get("dram_preloads", []), "data"),
        golden_result=_entries_to_regions(data.get("dram_checks", []), "expected"),
    )
