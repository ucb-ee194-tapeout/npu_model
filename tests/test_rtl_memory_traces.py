"""Replay actual LSU/XLU row ports with synchronous on-chip memory responses."""
import json
from pathlib import Path
from unittest.mock import Mock

import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import VLOAD, VSTORE, LW, SW, VTRPOSE_XLU
from npu_model.hardware.lsu import LoadStoreUnit
from npu_model.hardware.xlu import CrossLaneExecutionUnit
from npu_model.logging.logger import Logger
from npu_model.software import m, x
from tests.test_vector_rtl_timing import vpu, tick


def test_concurrent_memory_rtl_trace(vpu):
    state = vpu.arch_state
    cfg = DefaultHardwareConfig()
    lsu = LoadStoreUnit('LSU', Mock(spec=Logger), state, config=cfg)
    xlu = CrossLaneExecutionUnit('XLU', Mock(spec=Logger), state, config=cfg)
    for bank in range(64):
        state.mrf[bank][:] = torch.tensor([(bank*17 + row*3 + lane) & 255 for row in range(32) for lane in range(32)],dtype=torch.uint8)
    for bank in range(3):
        data = torch.tensor([(bank*23+row*5+lane)&255 for row in range(32) for lane in range(32)],dtype=torch.uint8)
        state.write_vmem(bank*262144, 0, data)
    state.write_xrf(1, 262144//4)
    state.write_xrf(2, 2*262144)
    state.write_xrf(3, 0xaabbccdd)
    commands = {1: VLOAD(vd=m(0),imm=0,rs1=x(0)),
                2: VSTORE(vd=m(2),imm=0,rs1=x(1)),
                3: LW(rd=x(10),imm=0,rs1=x(2)),
                10: SW(rs2=x(3),imm=4,rs1=x(2))}
    accesses, vr, vw, sw, results = [], [], [], [], []
    original_mreg = state.conflict_checker.access_mreg
    original_read, original_write, original_xrf = state.read_vmem, state.write_vmem, state.write_xrf
    def record(cycle,bank,row,write,owner):
        original_mreg(cycle,bank,row,write,owner)
        accesses.append((bank,row,write))
    def read(addr,imm,size):
        vr.append([addr//262144,(addr%262144)//32])
        return original_read(addr,imm,size)
    def write(addr,imm,data):
        raw = bytes(data.tolist())
        if len(raw)==32:
            vw.append([addr//262144,(addr%262144)//32,format(int.from_bytes(raw,'little'),'x')])
        else:
            offset = addr%32
            sw.append([addr//262144,(addr%262144)//32,format(int.from_bytes(raw,'little') << (8*offset),'x'),((1 << len(raw))-1) << offset])
        original_write(addr,imm,data)
    def write_xrf(reg,value):
        results.append(value)
        original_xrf(reg,value)
    state.conflict_checker.access_mreg = record
    state.read_vmem, state.write_vmem, state.write_xrf = read, write, write_xrf
    trace = json.loads((Path(__file__).parent / 'rtl/memory_traces.json').read_text())
    for expected in trace:
        for events in (accesses,vr,vw,sw,results): events.clear()
        cycle = expected['cycle']
        tick(lsu,commands.get(cycle))
        tick(xlu,VTRPOSE_XLU(vd=m(6),vs1=m(4)) if cycle==1 else None)
        reads = [[b,r] for b,r,w in accesses if not w]
        writes = [[b,r,format(int.from_bytes(bytes(state.mrf[b][r*32:(r+1)*32].tolist()),'little'),'x')] for b,r,w in accesses if w]
        actual = dict(cycle=cycle,reads=reads,writes=writes,vreads=vr,vwrites=vw,swrites=sw,result=results[0] if results else None)
        assert actual == expected, cycle
