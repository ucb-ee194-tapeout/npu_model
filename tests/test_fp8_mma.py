"""configs/fp8_mma.py against RTX 5090 tensor-core outputs (baremetal/generators/fp8_silicon)."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from npu_model.configs import fp8_mma

VECTORS = Path(__file__).resolve().parents[2] / "baremetal/generators/fp8_silicon"

# (name, {k: (a_k, b_k)} as E4M3 codes, C, RTX 5090 d) from fp8_silicon/directed.vec.gz
RULES = [
    ("F = 25: 1 - 1 + 2^-25", {0: (0x38, 0x38), 1: (0x38, 0xB8)}, 0x3300_0000, 0x3300_0000),
    ("F = 25: 1 - 1 + 2^-26", {0: (0x38, 0x38), 1: (0x38, 0xB8)}, 0x3280_0000, 0),
    ("unnormalized product: 2.25 - 2.25 + 2^-25", {0: (0x3C, 0x3C), 1: (0x3C, 0xBC)}, 0x3300_0000, 0x3300_0000),
    ("cut toward zero: 2^20 - 1.125^2", {0: (0x39, 0xB9)}, 0x4980_0000, 0x497F_FFEC),
    ("C is cut: 16 - (1 + 2^-23)", {0: (0x48, 0x48)}, 0xBF80_0001, 0x4170_0000),
    ("RZ output: 1 + 0.75 ulp", {0: (0x38, 0x38)}, 0x33C0_0000, 0x3F80_0000),
    ("RZ output: 1 + 1.5 ulp", {0: (0x38, 0x38)}, 0x3440_0000, 0x3F80_0001),
    ("one 32-element sum: a16 b16 = -1", {0: (0x38, 0x38), 16: (0x38, 0xB8)}, 0x3380_0000, 0x3380_0000),
    ("zero product does not set emax", {0: (0x00, 0x7E), 1: (0x38, 0x88)}, 0x3C80_0008, 0x3280_0000),
    ("subnormal input: e = -6", {0: (0x01, 0x38)}, 0xBAFF_FFFF, 0x3000_0000),
    ("all zero: d = C", {}, 0x8000_0003, 0x8000_0003),
    ("all zero: -0 + -0 products = +0", {k: (0x80, 0x00) for k in range(32)}, 0x8000_0000, 0),
    ("exact zero is +0", {0: (0x38, 0x38), 1: (0x38, 0xB8)}, 0xB080_0000, 0),
    ("E4M3 NaN", {0: (0x7F, 0x38)}, 0x3F80_0000, 0x7FFF_FFFF),
    ("C = +Inf", {0: (0x7E, 0xFE)}, 0x7F80_0000, 0x7F80_0000),
    ("C = NaN", {0: (0x38, 0x38)}, 0x7FC0_0000, 0x7FFF_FFFF),
]


def _read_vec(path):
    """a, b, c and the rtx5090 column of an fp8-vectors-v1 file."""
    raw = gzip.decompress(path.read_bytes())
    nl = raw.index(b"\n")
    header = json.loads(raw[:nl])
    n, body = header["count"], np.frombuffer(raw, np.uint8, offset=nl + 1)
    rec = body[:68 * n].reshape(n, 68)
    k = header["columns"].index("rtx5090")
    out = body[68 * n + 4 * n * k:68 * n + 4 * n * (k + 1)].view("<u4")
    return rec[:, :32], rec[:, 32:64], rec[:, 64:].copy().view("<u4")[:, 0], out


@pytest.mark.parametrize("name,pairs,c,d", RULES, ids=[r[0] for r in RULES])
def test_rule(name, pairs, c, d):
    a, b = np.zeros((2, 32), np.uint8)
    for k, (x, y) in pairs.items():
        a[k], b[k] = x, y
    assert int(fp8_mma.dot(a, b, c)) == d


@pytest.mark.parametrize("path", sorted(VECTORS.glob("*.vec.gz")), ids=lambda p: p.name.split(".")[0])
def test_silicon_vectors(path):
    a, b, c, rtx5090 = _read_vec(path)
    np.testing.assert_array_equal(fp8_mma.dot(a, b, c), rtx5090)


def test_matmul_is_dot_per_element():
    rng = np.random.default_rng(0)
    a, w = rng.integers(0, 256, (2, 32, 32), dtype=np.uint8)
    a[a & 0x7F == 0x7F] = 0x7E
    w[w & 0x7F == 0x7F] = 0x7E
    c = rng.normal(0, 1000, (32, 32)).astype(np.float32)
    acc = fp8_mma.matmul(a, w, c.view(np.uint32))
    assert acc.shape == (32, 32) and acc.dtype == np.uint32
    for r, j in [(0, 0), (0, 31), (31, 0), (5, 17), (31, 31)]:
        assert acc[r, j] == fp8_mma.dot(a[r], w[j], c.view(np.uint32)[r, j])
    # acc = c + A @ W^T within the cuts: 33 terms below 2^(emax - 25), and the RZ output
    x, y = _e4m3(a), _e4m3(w)
    bound = 2.0 ** -13 * (np.abs(c) + np.abs(x) @ np.abs(y).T)
    assert (np.abs(acc.view(np.float32) - (c + x @ y.T)) <= bound).all()


def _e4m3(codes):
    field, m = (codes >> 3) & 15, codes & 7
    mag = np.where(field > 0, (8 + m) * 2.0 ** (field.astype(np.int64) - 10), m * 2.0 ** -9)
    return np.where(codes >> 7 == 1, -mag, mag)
