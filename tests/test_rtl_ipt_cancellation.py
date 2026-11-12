"""IPT sums that cancel to fewer than eight bits, as observed on penguin26 silicon and in RTL simulation."""
import pytest
import torch

from npu_model.configs.rtl_math import ipt_row

# Two 32-term passes from a SmolVLA qk attention matmul, each seeded with the previous pass's BF16 partial.
A46 = [219, 180, 224, 235, 100, 91, 110, 113, 241, 96, 200, 104, 68, 231, 240, 110, 225, 81, 234, 233, 103, 105, 100, 222, 236, 229, 241, 109, 236, 97, 84, 104]
W46 = [93, 191, 104, 220, 112, 105, 219, 198, 236, 75, 95, 233, 222, 234, 236, 109, 105, 217, 91, 108, 233, 110, 84, 227, 89, 241, 84, 49, 91, 81, 79, 178]
A47 = [213, 238, 234, 193, 228, 196, 218, 111, 230, 116, 109, 112, 233, 79, 102, 220, 101, 112, 233, 178, 74, 76, 205, 233, 232, 91, 230, 104, 236, 220, 236, 102]
W47 = [93, 224, 225, 84, 234, 231, 240, 212, 229, 193, 114, 104, 218, 240, 96, 104, 223, 217, 82, 83, 100, 220, 232, 106, 233, 227, 232, 225, 76, 234, 221, 235]


@pytest.mark.parametrize('a, w, partial, expected', [
    (A46, W46, 0xc717, 0xbdc0),  # sum of -48 anchor units (six bits); was 0xbdb0
    (A47, W47, 0xc702, 0x3e00),  # sum of +64 anchor units (seven bits); was 0x3e40
])
def test_ipt_normalizes_sums_below_eight_bits(a, w, partial, expected):
    a = torch.tensor(a, dtype=torch.uint8).view(torch.float8_e4m3fn)
    w = torch.tensor([w], dtype=torch.uint8).view(torch.float8_e4m3fn)
    partial = torch.tensor([partial], dtype=torch.uint16).view(torch.bfloat16)
    assert ipt_row(a, w, partial).view(torch.uint16).item() == expected
