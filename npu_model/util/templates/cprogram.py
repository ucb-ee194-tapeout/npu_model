from torch._tensor_str import printoptions # type: ignore — this is annoyingly not publicly exported
import struct
import torch

def _memory_to_dict(
    memory_regions: list[tuple[int, torch.Tensor]],
    word_bytes: int = 4,  # uint32 based on C template
) -> dict[int, int]:
    result: dict[int, int] = {}
    fmt = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[word_bytes]

    for base, arr in memory_regions:
        raw = arr.flatten().contiguous().view(torch.uint8).numpy().tobytes()
        remainder = len(raw) % word_bytes
        if remainder:
            raw += b'\x00' * (word_bytes - remainder)
        n_words = len(raw) // word_bytes
        word_base = base
        for i, w in enumerate(struct.unpack_from(f'<{n_words}{fmt}', raw)):
            result[word_base + (i*4)] = w

    return result


def _fmt_array(words: list[int], name: str, words_per_line: int = 8, ctype: str="uint32_t", indent: str="    "):
    """Format a C array initializer."""
    lines: list[str] = []
    for i in range(0, len(words), words_per_line):
        chunk = words[i:i+words_per_line]
        vals = ", ".join(f"0x{w & 0xFFFFFFFF:08X}" for w in chunk)
        lines.append(f"{indent}{vals},")
    body = "\n".join(lines)
    return f"static const {ctype} {name}[] = {{{f'\n{body}\n' if len(body) != 0 else ''}}};"


def format_c_header_file(program_name: str, file_name: str, timeout: int | None, instructions: list[int], mem_regions: list[tuple[int, torch.Tensor]], golden_result: list[tuple[int, torch.Tensor]]):
    memory_dict  = _memory_to_dict(mem_regions)
    memory_size  = len(memory_dict)
    memory_addrs = list(memory_dict.keys())
    memory_vals  = list(memory_dict.values())

    result_dict  = _memory_to_dict(golden_result)
    result_size  = len(result_dict)
    result_addrs = list(result_dict.keys())
    result_vals  = list(result_dict.values())

    print(memory_dict)
    return f"""// Automatically generated from {file_name} by assembly.py
#ifndef {program_name}
#define {program_name}

#include <stdio.h>
#include <stdint.h>

#define TEST_NAME {program_name}
#define PROGRAM_TIMEOUT {timeout}

#define ATLAS_PROGRAM_LEN {len(instructions)}
{_fmt_array(instructions, "atlas_program", words_per_line = 1)}

#define PRELOAD_WORDS {memory_size}
{_fmt_array(memory_addrs, "preload_addrs")}
{_fmt_array(memory_vals, "preload_data")}

#define CHECK_WORDS {result_size}
{_fmt_array(result_addrs, "check_addrs")}
{_fmt_array(result_vals, "check_expected")}

#endif
"""