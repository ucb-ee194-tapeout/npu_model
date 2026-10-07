"""Cycle-by-cycle replay of actual VectorEngineTop/Verilator port traces."""
import json
from pathlib import Path

import pytest
import torch

from io import StringIO
from npu_model.util.converter import stream_to_instrs
from tests.test_vector_rtl_timing import vpu, tick

TRACES = json.loads((Path(__file__).parent / 'rtl/vector_traces.json').read_text())
MNEMONICS = dict(zip(
    'add sub mul rcp sqrt sin cos tanh log exp exp2 square cube rsum csum fp8pack fp8unpack relu rmax rmin cmax cmin pairmax pairmin mov vliOne vliCol vliRow vliAll'.split(),
    'vadd.bf16 vsub.bf16 vmul.bf16 vrecip.bf16 vsqrt.bf16 vsin.bf16 vcos.bf16 vtanh.bf16 vlog2.bf16 vexp.bf16 vexp2.bf16 vsquare.bf16 vcube.bf16 vredsum.row.bf16 vredsum.bf16 vpack.bf16.fp8 vunpack.fp8.bf16 vrelu.bf16 vredmax.row.bf16 vredmin.row.bf16 vredmax.bf16 vredmin.bf16 vmaximum.bf16 vminimum.bf16 vmov vli.one vli.col vli.row vli.all'.split()))

@pytest.mark.parametrize('name', TRACES)
def test_vector_rtl_port_trace(vpu, name):
    state = vpu.arch_state
    case = name
    name = name.split("_")[0]
    for bank in range(64):
        state.mrf[bank][:] = torch.tensor([
            ([0,0x8000,1,0x8001,0x007f,0x807f,0x7f80,0xff80,0x7fc1,0xffa1,0x7f7f,0xff7f,0x3f80,0xbf80,0x3c80,0x43ef][(bank*3+row+lane)%16] if case.endswith("_special") else 0x3e80 + bank * 32 + row * 4 + lane)
            for row in range(32) for lane in range(16)
        ], dtype=torch.uint16).view(torch.uint8)
    state.write_erf(0, 127)
    mnemonic = MNEMONICS[name]
    args = ('m4, 0x3f80' if name.startswith('vli') else
            'm4, m2, e0' if name in {'fp8pack', 'fp8unpack'} else
            'm4, m0, m2' if name in {'add','sub','mul','pairmax','pairmin'} else 'm4, m0')
    if case == 'add_mirrored': args = 'm4, m0, m0'
    if case == 'mov_inplace': args = 'm0, m0'
    insn = stream_to_instrs(StringIO(mnemonic + ' ' + args))[0]
    accesses = []
    original = state.conflict_checker.access_mreg
    def record(cycle, bank, row, write, owner):
        original(cycle, bank, row, write, owner)
        accesses.append((bank, row, write))
    state.conflict_checker.access_mreg = record
    for expected in TRACES[case]:
        accesses.clear()
        assert vpu.has_in_flight == expected['busy'], (case, expected['cycle'], 'busy')
        assert vpu.issue_busy_mask == expected['issue_busy'], (case, expected['cycle'], 'issue_busy')
        command = insn if expected['cycle'] == 1 else None
        if case == 'mov_overlap' and expected['cycle'] == 2:
            command = stream_to_instrs(StringIO('vrecip.bf16 m10, m8'))[0]
        if case == 'mov_handoff' and expected['cycle'] == 66:
            command = stream_to_instrs(StringIO('vmov m10, m8'))[0]
        tick(vpu, command)
        reads = [[b,r] for b,r,w in accesses if not w]
        writes = [[b,r, format(int.from_bytes(bytes(state.mrf[b][r*32:(r+1)*32].tolist()), 'little'), 'x')]
                  for b,r,w in accesses if w]
        assert sorted(reads) == sorted(expected['reads']), (name, expected['cycle'], 'reads')
        assert sorted(writes) == sorted(expected['writes']), (name, expected['cycle'], 'writes')
