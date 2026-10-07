"""Replay actual SA/IPT ports, arithmetic, accumulator chaining and overlap."""
import json
from pathlib import Path
from io import StringIO

import torch

from npu_model.util.converter import stream_to_instrs
from tests.test_mxu_rtl_timing import mxu, tick


def test_matrix_rtl_port_trace(mxu):
    name = 'sa' if mxu.mxu == 'mxu0' else 'ipt'
    trace = json.loads((Path(__file__).parent / f'rtl/{name}_traces.json').read_text())
    state = mxu.arch_state
    for bank in range(64):
        if bank in (2, 3):
            raw = torch.tensor([0x3d00 + bank*64 + row*4 + lane for row in range(32) for lane in range(16)], dtype=torch.uint16).view(torch.uint8)
        else:
            raw = torch.tensor([0x20 + (bank*7+row*3+lane)%48 + (128 if (row+lane)%3==0 else 0) for row in range(32) for lane in range(32)], dtype=torch.uint8)
        state.mrf[bank][:] = raw
    instructions = {}
    for cycle, op, bank, acc, weight in trace['commands']:
        assembly = [f'vmatpush.weight.{mxu.mxu} w{weight}, m{bank}',
                    f'vmatpush.acc.fp8.{mxu.mxu} acc{acc}, m{bank}',
                    f'vmatpush.acc.bf16.{mxu.mxu} acc{acc}, m{bank}',
                    f'vmatpop.fp8.acc.{mxu.mxu} m{bank}, acc{acc}, e0',
                    f'vmatpop.bf16.acc.{mxu.mxu} m{bank}, acc{acc}',
                    f'vmatmul.{mxu.mxu} acc{acc}, m{bank}, w{weight}',
                    f'vmatmul.acc.{mxu.mxu} acc{acc}, m{bank}, w{weight}'][op]
        instructions[cycle] = stream_to_instrs(StringIO(assembly))[0]
    state.write_erf(0, 127)
    accesses = []
    original = state.conflict_checker.access_mreg
    def record(cycle, bank, row, write, owner):
        original(cycle, bank, row, write, owner)
        accesses.append((bank, row, write))
    state.conflict_checker.access_mreg = record
    for expected in trace['events']:
        accesses.clear()
        tick(mxu, instructions.get(expected['cycle']))
        reads = [[b,r] for b,r,w in accesses if not w]
        writes = [[b,r, format(int.from_bytes(bytes(state.mrf[b][r*32:(r+1)*32].tolist()),'little'),'x')] for b,r,w in accesses if w]
        assert sorted(reads) == sorted(expected['reads']), (name,expected['cycle'],'reads')
        assert sorted(writes) == sorted(expected['writes']), (name,expected['cycle'],'writes')
