"""Program representation and parser for functional assembly (``.fs``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TextIO

from npu_model.configs import isa_definition as _isa_definition  # noqa: F401
from npu_model.configs.isa_definition import DMA_CONFIG_CH0, DMA_LOAD_CH0, DMA_STORE_CH0
from npu_model.isa import Instruction, IsaSpec, SBType, UJType
from npu_model.isa_patterns import Bundled
from npu_model.isa_types import (
    BoundedInt,
    Imm21,
    MatrixReg,
    Named,
    RegBase,
    SBImm12,
    ScalarReg,
)


class FunctionalAssemblyError(ValueError):
    """An ``.fs`` syntax or instruction-set error with source coordinates."""

    def __init__(self, message: str, line: int, column: int | None = None):
        self.line = line
        self.column = column
        where = f"line {line}" + (f", column {column}" if column is not None else "")
        super().__init__(f"{where}: {message}")


@dataclass(frozen=True)
class FunctionalInstruction:
    pc: int
    instruction: Instruction
    line: int
    source: str


@dataclass
class FunctionalProgram:
    instructions: list[FunctionalInstruction]
    labels: dict[str, int] = field(default_factory=dict)
    register_ids: dict[type[RegBase], dict[str, int]] = field(default_factory=dict)


_VIRTUAL_REGISTER = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\.([A-Za-z_][A-Za-z0-9_.]*)$")
_BUNDLE = re.compile(r"^(.+)\(([^()]*)\)$")
_LABEL = Named.is_label

# Functional DMA has no channel operand.  These canonical implementations
# share the existing architectural data-movement semantics; channel choice is
# intentionally absent from the functional state.
_FUNCTIONAL_ALIASES: dict[str, type[Instruction]] = {
    "dma.load": DMA_LOAD_CH0,
    "dma.store": DMA_STORE_CH0,
    "dma.config": DMA_CONFIG_CH0,
}

_FORBIDDEN = {
    "delay",
    "ecall",
    "ebreak",
    "csrrw",
    "csrrs",
    "csrrc",
    "csrrwi",
    "csrrsi",
    "csrrci",
}


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _tokens(line: str) -> list[str]:
    return [token.rstrip(",") for token in re.split(r"[\s,]+", line) if token]


def _register_index(
    register_type: type[RegBase],
    token: str,
    register_ids: dict[type[RegBase], dict[str, int]],
) -> int:
    match = _VIRTUAL_REGISTER.fullmatch(token)
    if not match or match.group(1) != register_type.fmt:
        raise ValueError(
            f"expected virtual {register_type.reg_name} like "
            f"'{register_type.fmt}.name', got '{token}'"
        )
    name = match.group(2)
    names = register_ids.setdefault(register_type, {})
    if name not in names:
        # The shared semantic helpers use adjacent MRF slots for BF16 pairs.
        # Give each virtual matrix name a disjoint two-slot backing window;
        # this is not physical allocation and has no architectural limit.
        stride = 2 if register_type is MatrixReg else 1
        # Start above zero because shared architectural write helpers reserve
        # physical x0.  This is only an internal virtual ID.
        base = 1 if register_type is ScalarReg else 0
        names[name] = base + len(names) * stride
    return names[name]


def _parse_value(
    field: Named,
    token: str,
    *,
    instruction: type[Instruction],
    pc: int,
    labels: dict[str, int],
    register_ids: dict[type[RegBase], dict[str, int]],
) -> BoundedInt | int:
    inner = field.inner
    if isinstance(inner, type) and issubclass(inner, RegBase):
        return _register_index(inner, token, register_ids)

    if issubclass(instruction, (SBType, UJType)) and field.repr == "imm":
        if token in labels:
            displacement = labels[token] - pc
        elif _LABEL.fullmatch(token):
            raise ValueError(f"undefined label '{token}'")
        else:
            displacement = int(token, 0)
        limit = 2048 if issubclass(instruction, SBType) else 524288
        if not -limit <= displacement < limit:
            raise ValueError(f"instruction-word offset {displacement} out of range")
        return (SBImm12 if issubclass(instruction, SBType) else Imm21)(displacement * 2)

    if field.label_support and token in labels:
        return inner(labels[token])
    if field.label_support and _LABEL.fullmatch(token):
        raise ValueError(f"undefined label '{token}'")
    return inner(token)


def _parse_param(
    param: Named,
    token: str,
    *,
    instruction: type[Instruction],
    pc: int,
    labels: dict[str, int],
    register_ids: dict[type[RegBase], dict[str, int]],
) -> dict[str, BoundedInt | int]:
    if isinstance(param, Bundled):
        match = _BUNDLE.fullmatch(token)
        if not match:
            raise ValueError(f"expected {param.format_arg()}, got '{token}'")
        immediate, register = match.groups()
        return {
            **_parse_param(
                param.imm,
                immediate,
                instruction=instruction,
                pc=pc,
                labels=labels,
                register_ids=register_ids,
            ),
            **_parse_param(
                param.reg,
                register,
                instruction=instruction,
                pc=pc,
                labels=labels,
                register_ids=register_ids,
            ),
        }
    return {
        param.repr: _parse_value(
            param,
            token,
            instruction=instruction,
            pc=pc,
            labels=labels,
            register_ids=register_ids,
        )
    }


def _functional_operations() -> dict[str, type[Instruction]]:
    operations: dict[str, type[Instruction]] = {}
    for mnemonic, instruction in IsaSpec.operations.items():
        if not getattr(instruction, "functional", True):
            continue
        operations[mnemonic] = instruction
    operations.update(_FUNCTIONAL_ALIASES)
    return operations


def parse_functional_assembly(source: TextIO | str) -> FunctionalProgram:
    """Parse the `.fs` subset using virtual typed register names.

    Virtual names look like ``x.address``, ``e.scale``, ``m.tile``,
    ``w.weights`` and ``acc.partial``.  They never encode physical indices.
    """
    if isinstance(source, str):
        raw_lines = source.splitlines()
    else:
        raw_lines = list(source)
    lines = [_strip_comment(line) for line in raw_lines]
    labels: dict[str, int] = {}
    entries: list[tuple[int, str, list[str]]] = []
    pc = 0
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if not _LABEL.fullmatch(label):
                raise FunctionalAssemblyError(f"invalid label '{label}'", line_number)
            if label in labels:
                raise FunctionalAssemblyError(f"duplicate label '{label}'", line_number)
            labels[label] = pc
            continue
        tokens = _tokens(line)
        if not tokens:
            continue
        entries.append((line_number, line, tokens))
        pc += 1

    operations = _functional_operations()
    register_ids: dict[type[RegBase], dict[str, int]] = {}
    instructions: list[FunctionalInstruction] = []
    pc = 0
    for line_number, source_line, tokens in entries:
        mnemonic = tokens[0].lower()
        if mnemonic in _FORBIDDEN:
            raise FunctionalAssemblyError(
                f"'{mnemonic}' is not part of the functional instruction subset",
                line_number,
            )
        if mnemonic.startswith("dma.wait") or mnemonic.startswith(
            ("dma.load.ch", "dma.store.ch", "dma.config.ch")
        ):
            raise FunctionalAssemblyError(
                f"'{mnemonic}' is channel-specific or a synchronization operation; .fs omits these",
                line_number,
            )
        instruction_type = operations.get(mnemonic)
        if instruction_type is None:
            raise FunctionalAssemblyError(f"unsupported functional mnemonic '{mnemonic}'", line_number)
        params = instruction_type.params
        if len(tokens) != len(params) + 1:
            raise FunctionalAssemblyError(
                f"'{mnemonic}' expects {len(params)} operand(s), got {len(tokens) - 1}",
                line_number,
            )
        kwargs: dict[str, BoundedInt | int] = {}
        try:
            for index, param in enumerate(params, start=1):
                kwargs.update(
                    _parse_param(
                        param,
                        tokens[index],
                        instruction=instruction_type,
                        pc=pc,
                        labels=labels,
                        register_ids=register_ids,
                    )
                )
            instruction = instruction_type(**kwargs)
        except (ValueError, TypeError, ExceptionGroup) as exc:
            raise FunctionalAssemblyError(str(exc), line_number) from exc
        instructions.append(FunctionalInstruction(pc, instruction, line_number, source_line))
        pc += 1
    return FunctionalProgram(instructions, labels, register_ids)


def load_functional_assembly(path: str | Path) -> FunctionalProgram:
    with open(path, encoding="utf-8") as source:
        return parse_functional_assembly(source)
