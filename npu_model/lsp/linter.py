"""
NPU Assembly Linter — core linting logic.

Validates .S files against the NPU ISA defined in npu_model, producing
per-line/column diagnostics suitable for use in an LSP server.
"""

import re
import sys
from typing import cast, Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make npu_model importable regardless of how the server process
# was launched.  The lint folder lives at:
#   <project_root>/npu_model/lint :3/
# so going up two levels gives us <project_root>, where `npu_model` lives.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import npu_model.configs.isa_definition as _isa_def  # noqa: F401, E402 # type: ignore 
from npu_model.isa import IsaSpec  # noqa: E402
from npu_model.isa_types import (
    AsmError,  # noqa: E402
    ScalarReg,  # noqa: E402
)

# ---------------------------------------------------------------------------
# Diagnostic model
# ---------------------------------------------------------------------------


class Severity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Diagnostic:
    line: int  # 0-indexed
    col_start: int
    col_end: int
    message: str
    severity: Severity = Severity.ERROR


# ---------------------------------------------------------------------------
# Tokeniser  (comment-stripping, position-preserving)
# ---------------------------------------------------------------------------

PosToken = tuple[str, int, int]  # (text, col_start, col_end)


def _strip_comment(raw: str) -> str:
    idx = raw.find("#")
    return raw[:idx] if idx >= 0 else raw


def _tokenize(raw: str) -> list[PosToken]:
    effective = _strip_comment(raw).replace(",", " ")
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", effective)]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

_VALID_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:$")
_COLON_AT_END_RE = re.compile(r":\s*$")

# Keep track of labels we ran into last time we linted.
SEEN_LABELS: dict[str, int] = {}  # name -> first definition line (0-indexed)

def _lint_label_def(stripped: str, lineno: int) -> list[Diagnostic]:
    if not _VALID_LABEL_RE.match(stripped):
        return [
            Diagnostic(
                lineno,
                0,
                len(stripped),
                f"Malformed label '{stripped[:-1]}' — labels must start with a letter or "
                f"'_' and contain only letters, digits, '_', or '.'",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Pseudo-instruction lint functions  (same style as InstructionPattern.lint)
# ---------------------------------------------------------------------------


def _lint_nop(tokens: list[str], labels: list[str]) -> list[AsmError]:
    if len(tokens) != 1:
        return [
            AsmError(
                f"'nop' takes no operands, got {len(tokens) - 1}",
                token_index=0,
            )
        ]
    return []


def _lint_li(tokens: list[str], labels: list[str]) -> list[AsmError]:
    if len(tokens) != 3:
        return [
            AsmError(
                f"'li' expects 2 operands ({ScalarReg.fmt}<rd> imm32), got {len(tokens) - 1}",
                token_index=0,
            )
        ]
    errors: list[AsmError] = []
    errors.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
    try:
        v = int(tokens[2], 0)
        if not (-(1 << 31) <= v <= 0xFFFF_FFFF):
            errors.append(
                AsmError(
                    f"'li' immediate {v} is out of 32-bit range",
                    token_index=2,
                )
            )
    except ValueError:
        errors.append(
            AsmError(
                f"Expected integer immediate for 'li', got '{tokens[2]}'",
                token_index=2,
            )
        )
    return errors


_PSEUDOS: dict[str, Callable[[list[str], list[str]], list[AsmError]]] = {
    "nop": _lint_nop,
    "li": _lint_li,
}

# ---------------------------------------------------------------------------
# AsmError -> Diagnostic conversion
# ---------------------------------------------------------------------------


def _to_diagnostic(err: AsmError, lineno: int, toks: list[PosToken]) -> Diagnostic:
    idx = err.token_index
    if idx < len(toks):
        cs, ce = toks[idx][1], toks[idx][2]
    else:
        cs, ce = toks[0][1], toks[-1][2]
    return Diagnostic(lineno, cs, ce, str(err))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lint_text(source: str) -> list[Diagnostic]:
    """
    Lint the full text of an NPU assembly file and return a (possibly empty)
    list of Diagnostics sorted by (line, col_start).
    """
    lines = source.splitlines()
    diags: list[Diagnostic] = []

    # ------------------------------------------------------------------
    # Pass 1 — collect label definitions; flag duplicates and bad names.
    # ------------------------------------------------------------------
    global SEEN_LABELS
    SEEN_LABELS.clear()

    for lineno, raw in enumerate(lines):
        stripped = _strip_comment(raw).strip()
        if not stripped or not _COLON_AT_END_RE.search(stripped):
            continue

        diags.extend(_lint_label_def(stripped, lineno))

        label_name = stripped.rstrip(":").strip()
        if label_name in SEEN_LABELS:
            diags.append(
                Diagnostic(
                    lineno,
                    0,
                    len(raw.rstrip()),
                    f"Duplicate label '{label_name}' (first defined on line "
                    f"{SEEN_LABELS[label_name] + 1})",
                    Severity.WARNING,
                )
            )
        else:
            SEEN_LABELS[label_name] = lineno

    labels = list(SEEN_LABELS.keys())

    # ------------------------------------------------------------------
    # Pass 2 — validate every instruction line.
    # ------------------------------------------------------------------
    for lineno, raw in enumerate(lines):
        stripped = _strip_comment(raw).strip()
        if not stripped or _COLON_AT_END_RE.search(stripped):
            continue

        toks = _tokenize(raw)
        if not toks:
            continue

        plain = [t for t, _, _ in toks]
        mnemonic = plain[0].lower()

        if mnemonic in _PSEUDOS:
            errors = _PSEUDOS[mnemonic](plain, labels)
        elif mnemonic in IsaSpec.operations:
            insn = IsaSpec.operations[mnemonic]
            # For the same reason we have to cast in assembler oops.
            errors = cast(Callable[[list[str],list[str]],list[AsmError]], insn.lint)(plain, labels) if hasattr(insn, 'lint') else []
        else:
            diags.append(
                Diagnostic(
                    lineno,
                    toks[0][1],
                    toks[0][2],
                    f"Unknown mnemonic '{plain[0]}'",
                )
            )
            continue

        for err in errors:
            diags.append(_to_diagnostic(err, lineno, toks))

    diags.sort(key=lambda d: (d.line, d.col_start))
    return diags
