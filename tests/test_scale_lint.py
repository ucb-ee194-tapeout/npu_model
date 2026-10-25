import io
from dataclasses import replace

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware.arch_state import ArchState
from npu_model.isa import IsaSpec
from npu_model.isa_types import EXU, Accumulator, Bundled, MatrixReg, WeightBuffer
from npu_model.scale_lint import data_writes, scale_warnings
from npu_model.util.converter import stream_to_instrs

STALE = "m{}'s data may be written after its last vload.scale (pc {})"
UNLOADED = "m{}'s scales may not be loaded since reset"

CASES = {
    "data, then scales": ("""
        vload m0, 0(x1)
        vload.scale m0, 4(x1)
        vmatmul.nv acc0, m0, w0
    """, []),
    "scales, then data": ("""
        vload.scale m0, 4(x1)
        vload m0, 0(x1)
        vmatmul.nv acc0, m0, w0
    """, [(2, STALE.format(0, 1))]),
    "never loaded": ("""
        vload m0, 0(x1)
        vmatpush.weight.nv w0, m0
    """, [(1, UNLOADED.format(0))]),
    "an FP32 pop writes four registers": ("""
        vload.scale m7, 4(x1)
        vload.scale m8, 8(x1)
        vmatpop.fp32.acc.nv m4, acc0
        vmatmul.nv acc0, m7, w0
        vmatmul.nv acc0, m8, w0
    """, [(3, STALE.format(7, 2))]),
    "a loop reloads the data in the delay slot": ("""
        vload m0, 0(x1)
        vload.scale m0, 4(x1)
    loop:
        vmatmul.acc.nv acc0, m0, w0
        bne x2, x0, loop
        vload m0, 0(x3)
        ecall
    """, [(2, STALE.format(0, 4))]),
    "a jump runs its delay slot and skips the code after it": ("""
        vload m0, 0(x1)
        jal x0, done
        vload.scale m0, 4(x1)
        vload m0, 0(x3)
    done:
        vmatmul.nv acc0, m0, w0
    """, []),
}


@pytest.mark.parametrize("case", CASES)
def test_scale_warnings(case: str) -> None:
    source, expected = CASES[case]
    assert scale_warnings(stream_to_instrs(io.StringIO(source))) == expected


def operand(named) -> int:
    """Distinct m-registers, slot and buffer 0, and 1 for every other operand."""
    if named.inner is MatrixReg:
        return {"vd": 4, "vs1": 8, "vs2": 12}[named.repr]
    return 0 if named.inner in (Accumulator, WeightBuffer) else 1


@pytest.mark.parametrize("mnemonic", sorted(
    m for m, cls in IsaSpec.operations.items() if cls.exu not in (EXU.SCALAR, EXU.DMA)))
def test_data_writes_match_the_model(mnemonic: str) -> None:
    """data_writes are the registers whose random contents exec changes."""
    cls = IsaSpec.operations[mnemonic]
    named = [n for p in cls.params for n in ((p.reg, p.imm) if isinstance(p, Bundled) else (p,))]
    insn = cls(**{n.repr: operand(n) for n in named})
    state = ArchState(replace(DefaultHardwareConfig.arch_state_config, randomize_init=True))
    before = [r.clone() for r in state.mrf]
    insn.exec(state)
    changed = [r for r in range(len(before)) if not torch.equal(before[r], state.mrf[r])]
    assert changed == list(data_writes(insn))
