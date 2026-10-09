"""Atomic functional interpreter for ``.fs`` programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .program import FunctionalAssemblyError, FunctionalProgram
from .state import FunctionalConfig, FunctionalState


@dataclass
class FunctionalResult:
    state: FunctionalState
    instructions_executed: int
    final_pc: int
    halt_reason: str | None

    def named_registers(self, program: FunctionalProgram) -> dict[str, dict[str, object]]:
        """Return values keyed by the virtual names used by ``program``."""
        output: dict[str, dict[str, object]] = {}
        for register_type, names in program.register_ids.items():
            file_name = register_type.fmt
            if file_name in {"w", "acc"}:
                for unit in ("mxu0", "mxu1"):
                    file = _register_file(self.state, file_name, unit)
                    output[f"{file_name}.{unit}"] = {
                        name: _copy_value(file[index]) for name, index in names.items()
                    }
            else:
                file = _register_file(self.state, file_name)
                output[file_name] = {
                    name: _copy_value(file[index]) for name, index in names.items()
                }
        return output


def _copy_value(value: object) -> object:
    return value.clone() if isinstance(value, torch.Tensor) else value


def _register_file(state: FunctionalState, prefix: str, unit: str = "mxu0"):
    if prefix == "x":
        return state.xrf
    if prefix == "e":
        return state.erf
    if prefix == "m":
        return state.mrf
    if prefix == "w":
        return state.wb[unit]
    if prefix == "acc":
        return state.acc[unit]
    raise ValueError(f"unknown virtual register class '{prefix}'")


class FunctionalInterpreter:
    """Execute instruction semantics without hardware timing or allocation."""

    def __init__(self, config: FunctionalConfig | None = None):
        self.state = FunctionalState(config)

    def run(
        self,
        program: FunctionalProgram,
        *,
        max_instructions: int = 1_000_000,
        dram: Mapping[int, torch.Tensor | bytes | bytearray] | None = None,
        vmem: Mapping[int, torch.Tensor | bytes | bytearray] | None = None,
    ) -> FunctionalResult:
        if max_instructions <= 0:
            raise ValueError("max_instructions must be positive")
        state = self.state
        state.reset()
        for offset, data in (dram or {}).items():
            state.load_dram(offset, data)
        for offset, data in (vmem or {}).items():
            state.load_vmem(offset, data)

        pc = 0
        executed = 0
        while 0 <= pc < len(program.instructions):
            if executed >= max_instructions:
                raise RuntimeError(
                    f"functional instruction limit ({max_instructions}) reached at PC {pc}"
                )
            entry = program.instructions[pc]
            state.execute_pc = pc
            state.npc = pc + 1
            state.redirect_requested = False
            try:
                entry.instruction.exec(state)
            except Exception as exc:
                raise FunctionalAssemblyError(
                    f"executing '{entry.instruction.mnemonic}' failed: {exc}", entry.line
                ) from exc
            executed += 1
            if state.halted:
                pc += 1
                break
            pc = state.npc if state.redirect_requested else pc + 1
        if pc < 0:
            raise RuntimeError(f"functional control flow reached invalid PC {pc}")
        if pc > len(program.instructions):
            raise RuntimeError(f"functional control flow reached invalid PC {pc}")
        return FunctionalResult(state, executed, pc, state.halt_reason)
