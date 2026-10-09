"""Machine-readable metadata for the Atlas assembly syntax and encodings.

This module describes the existing ISA classes; it is deliberately not a
second instruction catalog.  Compiler tooling can call :func:`export_isa` or
serialize its result as JSON.
"""

from __future__ import annotations

from typing import Any

from .isa import (
    CSRType,
    IType,
    IsaSpec,
    RType,
    SBType,
    SType,
    UJType,
    UType,
    VIType,
    VLSType,
    VRType,
)
from .isa_types import BoundedInt
from .isa_patterns import Bundled

SCHEMA_VERSION = 1

_FORMAT_FIELDS: dict[type, list[str]] = {
    RType: ["funct7", "rs2", "rs1", "funct3", "rd", "opcode"],
    IType: ["imm", "rs1", "funct3", "rd", "opcode"],
    SType: ["imm[11:5]", "rs2", "rs1", "funct3", "imm[4:0]", "opcode"],
    SBType: ["imm[12]", "imm[10:5]", "rs2", "rs1", "funct3", "imm[4:1]", "imm[11]", "opcode"],
    UType: ["imm[19:0]", "rd", "opcode"],
    UJType: ["imm[20]", "imm[10:1]", "imm[11]", "imm[19:12]", "rd", "opcode"],
    VLSType: ["imm", "rs1", "funct2", "vd", "opcode"],
    VRType: ["funct7", "vs2", "vs1/es1", "vd", "opcode"],
    VIType: ["imm", "funct3", "vd", "opcode"],
    CSRType: ["imm", "rs1", "funct3", "rd", "opcode"],
}


def _bounded_type(typ: type) -> bool:
    return isinstance(typ, type) and issubclass(typ, BoundedInt)


def _type_info(typ: type) -> dict[str, Any]:
    info: dict[str, Any] = {"name": typ.__name__}
    if _bounded_type(typ):
        info.update(
            lower_bound=typ.lower_bound,
            upper_bound=typ.upper_bound,
            format=typ.fmt,
        )
        if typ.upper_bound > 1:
            info["bit_width"] = (typ.upper_bound - 1).bit_length()
        if getattr(typ, "fmt", "") in {"x", "e", "m", "w", "acc"}:
            info["kind"] = "register"
        else:
            info["kind"] = "immediate"
    return info


def _operand_access(mnemonic: str, name: str) -> str:
    """Return an instruction operand's dataflow role where it is knowable."""
    if name == "imm":
        return "immediate"
    if mnemonic.startswith(("dma.load", "dma.store")):
        return "read"
    if mnemonic.startswith("dma.config"):
        return "read"
    if mnemonic.startswith(("beq", "bne", "blt", "bge")):
        return "read"
    if mnemonic in {"sb", "sh", "sw"}:
        return "read"
    if mnemonic in {"lb", "lh", "lw", "lbu", "lhu", "seld"} and name == "rs1":
        return "read"
    if mnemonic.startswith("csr") and name == "rs1" and mnemonic.endswith("i"):
        return "immediate"
    if mnemonic == "vstore":
        return "read" if name in {"vd", "rs1", "imm"} else "unknown"
    if mnemonic.startswith("vmatmul.acc") and name == "vd":
        return "readwrite"
    if name in {"rd", "vd"}:
        return "write"
    if name in {"rs1", "rs2", "vs1", "vs2", "es1"}:
        return "read"
    return "unknown"


def _format_type(instruction: type) -> type | None:
    for parent in instruction.__mro__:
        if parent in _FORMAT_FIELDS:
            return parent
    return None


def _param_info(param: Any, position: int, mnemonic: str) -> dict[str, Any]:
    if isinstance(param, Bundled):
        fields = [param.imm, param.reg]
        syntax = param.format_arg()
    else:
        fields = [param]
        syntax = param.format_arg()
    return {
        "position": position,
        "syntax": syntax,
        "fields": [
            {
                "name": field.repr,
                "type": _type_info(field.inner),
                "access": _operand_access(mnemonic, field.repr),
                "label_allowed": bool(field.label_support),
            }
            for field in fields
        ],
    }


def instruction_metadata(mnemonic: str, instruction: type) -> dict[str, Any]:
    """Describe one registered true-assembly instruction."""
    fmt = _format_type(instruction)
    encoding: dict[str, int] = {}
    for field in ("opcode", "funct2", "funct3", "funct7"):
        value = getattr(instruction, field, None)
        if isinstance(value, int):
            encoding[field] = int(value)
    return {
        "mnemonic": mnemonic,
        "instruction_class": instruction.__name__,
        "functional_supported": bool(getattr(instruction, "functional", True)),
        "assembly_operands_are_ordered": True,
        "format": fmt.__name__ if fmt else None,
        "operands": [
            _param_info(param, index, mnemonic)
            for index, param in enumerate(instruction.params, start=1)
        ],
        "encoding": {
            **encoding,
            "fields_msb_to_lsb": list(_FORMAT_FIELDS.get(fmt, [])),
        },
    }


def export_isa() -> dict[str, Any]:
    """Return a JSON-serializable description of the true assembly ISA."""
    # Instruction classes register themselves when the canonical definition
    # module is imported.  Import here so standalone compiler tools get a
    # complete manifest from this public API.
    from .configs import isa_definition as _isa_definition  # noqa: F401

    instructions = [
        instruction_metadata(mnemonic, instruction)
        for mnemonic, instruction in sorted(IsaSpec.operations.items())
    ]
    from .functional.program import _functional_operations

    functional_instructions = []
    for mnemonic, instruction in sorted(_functional_operations().items()):
        item = instruction_metadata(mnemonic, instruction)
        item["functional_supported"] = True
        # Functional operations do not have binary encodings of their own.
        # In particular, channel-free DMA forms are semantic aliases, not CH0.
        item.pop("encoding")
        if mnemonic in {"dma.load", "dma.store", "dma.config"}:
            item["true_assembly_mnemonics"] = [
                f"{mnemonic}.ch{channel}" for channel in range(8)
            ]
        functional_instructions.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "virtual_register_syntax": {
            "scalar": "x.<name>",
            "exponent": "e.<name>",
            "matrix": "m.<name>",
            "weight_buffer": "w.<name>",
            "accumulator": "acc.<name>",
        },
        "instructions": instructions,
        "functional_assembly": {
            "extension": ".fs",
            "instructions": functional_instructions,
            "excluded": [
                "delay",
                "dma.wait.ch<N>",
                "dma.load/store/config.ch<N> (channel-free spellings are used)",
                "CSR/system instructions in the initial subset",
            ],
            "dma_spellings": {
                "dma.load": [f"dma.load.ch{channel}" for channel in range(8)],
                "dma.store": [f"dma.store.ch{channel}" for channel in range(8)],
                "dma.config": [f"dma.config.ch{channel}" for channel in range(8)],
            },
        },
    }
