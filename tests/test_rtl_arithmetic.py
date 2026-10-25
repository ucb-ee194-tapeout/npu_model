"""Independent RTL outputs for default MXU arithmetic, including special bits."""
import json
from pathlib import Path

import torch

from npu_model.configs.fp8_mma import dot
from npu_model.configs.rtl_math import pack_row, unpack_row


def test_mxu_arithmetic_encodings_against_rtl():
    cases = json.loads((Path(__file__).parent / 'rtl/arithmetic.json').read_text())
    for i, case in enumerate(cases):
        a = torch.tensor(case['a'],dtype=torch.uint8).view(torch.float8_e4m3fn)
        p = torch.tensor([case['partial']],dtype=torch.uint16).view(torch.bfloat16)
        assert dot(case['a'],case['w'],case['c']) == case['ipt'], (i,'ipt')
        assert pack_row(p,case['scale']).item() == case['quant'], (i,'quant')
        assert unpack_row(a.view(torch.uint8)[:1]).view(torch.uint16).item() == case['dequant'], (i,'dequant')
