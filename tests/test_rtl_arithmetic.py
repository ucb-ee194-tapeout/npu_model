"""Independent RTL outputs for default MXU arithmetic, including special bits."""
import json
from pathlib import Path

from npu_model.configs.fp8_mma import dot


def test_mxu_arithmetic_encodings_against_rtl():
    cases = json.loads((Path(__file__).parent / 'rtl/arithmetic.json').read_text())
    for i, case in enumerate(cases):
        assert dot(case['a'],case['w'],case['c']) == case['ipt'], (i,'ipt')
