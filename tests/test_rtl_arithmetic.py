"""Independent RTL outputs for default MXU arithmetic, including special bits."""
import json
from pathlib import Path

import torch

from npu_model.hardware.rtl_math import fma_bits, ipt_row
from npu_model.hardware.vpu import pack_row, unpack_row


def test_mxu_arithmetic_encodings_against_rtl():
    cases = json.loads((Path(__file__).parent / 'rtl/arithmetic.json').read_text())
    for i, case in enumerate(cases):
        a = torch.tensor(case['a'],dtype=torch.uint8).view(torch.float8_e4m3fn)
        w = torch.tensor([case['w']],dtype=torch.uint8).view(torch.float8_e4m3fn)
        p = torch.tensor([case['partial']],dtype=torch.uint16).view(torch.bfloat16)
        assert fma_bits(case['a'][0],case['w'][0],case['partial']) == case['fma'], (i,'fma')
        assert ipt_row(a,w,p).view(torch.uint16).item() == case['ipt'], (i,'ipt')
        assert pack_row(p,case['scale'],mxu=True).item() == case['quant'], (i,'quant')
        assert unpack_row(a.view(torch.uint8)[:1],127).view(torch.uint16).item() == case['dequant'], (i,'dequant')
