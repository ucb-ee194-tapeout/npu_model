"""The row count M (vsetm): an M-row instruction reads and writes rows 0..M-1 of its registers and
accumulators, and the other rows keep their values. vpack, vunpack, vtrpose, the weight pushes and
the scale transfers take all rows."""
from io import StringIO

import pytest
import torch

from npu_model.isa import TABLE
from npu_model.software.instruction import Uop
from npu_model.util.converter import stream_to_instrs
from tests.test_exec_units import asm, make

COLUMN_REDUCTIONS = {"vredsum.bf16": 0x8000, "vredmax.bf16": 0xFF80, "vredmin.bf16": 0x7F80}   # identities
NV_OPERANDS = {"vmatpush.weight.nv": "w1, m8", "vmatpush.acc.bf16.nv": "acc1, m2", "vmatpush.acc.fp32.nv": "acc1, m4",
               "vmatpop.bf16.acc.nv": "m4, acc1", "vmatpop.bf16.scaled.acc.nv": "m4, acc1, e0",
               "vmatpop.fp32.acc.nv": "m12, acc1", "vmatmul.nv": "acc1, m8, w1", "vmatmul.acc.nv": "acc1, m8, w1"}


def text(name: str) -> str:
    if name in NV_OPERANDS:
        return f"{name} {NV_OPERANDS[name]}"
    if name in {"vload", "vstore", "vload.scale", "vstore.scale"}:
        return f"{name} m6, 8(x2)"
    return asm(name)


def tensor_instructions(m: str) -> list[str]:
    return sorted(n for n, e in TABLE.entries.items() if e.m == m)


def run(name: str, rows: int, prepare=None):
    """The state after exec of `name` at M = rows, from the same start as every other run."""
    state = make().arch_state
    if prepare:
        prepare(state)
    state.rows = rows
    stream_to_instrs(StringIO(text(name)))[0].exec(state)
    return state


def mregs(state) -> torch.Tensor:
    return torch.stack([state.read_mrf_u8(r) for r in range(len(state.mrf))])


def accs(state) -> torch.Tensor:
    return torch.stack([state.acc["mxu1"][a].view(torch.int32) for a in range(2)] +
                       [torch.tensor(state.acc_nv[a], dtype=torch.int64).to(torch.int32) for a in range(2)])


def test_vsetm_sets_m_and_a_start_sets_32():
    state = make().arch_state
    assert state.rows == 32
    for m in (1, 7, 32):
        stream_to_instrs(StringIO(f"vsetm {m}"))[0].exec(state)
        assert state.rows == m
    state.reset()
    assert state.rows == 32
    with pytest.raises(Exception):
        stream_to_instrs(StringIO("vsetm 0"))
    with pytest.raises(Exception):
        stream_to_instrs(StringIO("vsetm 33"))


@pytest.mark.parametrize("m", [1, 3, 31])
@pytest.mark.parametrize("name", tensor_instructions("rows"))
def test_an_m_row_instruction_moves_rows_below_m(name, m):
    """Rows 0..M-1 equal the 32-row instruction (a column reduction's, with rows M..31 of its source
    pair set to the identity of the reduction); the other rows, and VMEM lines M.., keep their values."""
    def identity(state):
        if name in COLUMN_REDUCTIONS:
            for reg in (0, 1):
                state.mrf[reg].view(torch.uint16).view(32, 16)[m:] = COLUMN_REDUCTIONS[name]
    start, full, got = make().arch_state, run(name, 32, identity), run(name, m)
    for b, f, g in ((mregs(start), mregs(full), mregs(got)), (accs(start), accs(full), accs(got))):
        assert torch.equal(g[:, :m], f[:, :m]), "rows 0..M-1"
        assert torch.equal(g[:, m:], b[:, m:]), "rows M.."
    lines, stored = start.vmem.view(-1, 32), full.vmem.view(-1, 32)
    changed = (stored != lines).any(dim=1).nonzero().flatten().tolist()
    first = changed[0] if changed else 0
    expected = lines.clone()
    expected[first:first + m] = stored[first:first + m]
    assert torch.equal(got.vmem.view(-1, 32), expected), "VMEM lines 0..M-1 of a vstore"


@pytest.mark.parametrize("name", tensor_instructions("all"))
def test_an_all_row_instruction_ignores_m(name):
    full, got = run(name, 32), run(name, 3)
    assert torch.equal(mregs(got), mregs(full)) and torch.equal(accs(got), accs(full))
    assert torch.equal(got.vmem, full.vmem)
    assert all(torch.equal(a, b) for a, b in zip(got.mrf_scale + got.wb_nv + got.wb_nv_scale,
                                                 full.mrf_scale + full.wb_nv + full.wb_nv_scale))
