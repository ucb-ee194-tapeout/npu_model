"""FP8 arithmetic of NVIDIA Blackwell tensor cores, the target of the Atlas FP8 MXU (mxu1).

One output element is d = C + a . b over 32 E4M3 pairs with an FP32 C, as one fused dot-add
(MMA-Sim "FDA" with F = 25, arXiv 2511.10909). It equals the RTX 5090 bit for bit (mma.sync
m16n8k32 E4M3 x E4M3 + FP32, SASS QMMA.16832.F32.E4M3.E4M3) on the vectors in
baremetal/generators/fp8_silicon:
  1. Any E4M3 NaN (S.1111.111) or a NaN C gives 0x7FFFFFFF. Else an infinite C gives C.
  2. Each product is exact, with an unnormalized significand: sig_a * sig_b / 64 x 2^(e_a + e_b),
     where sig = 8 + m and e = E - 7 for a normal E4M3, and sig = m and e = -6 for a subnormal.
  3. emax is the largest exponent of C (-126 for a zero or subnormal C) and of the nonzero
     products. Zero products do not set emax.
  4. Every nonzero product and C is cut toward zero (magnitude truncation) at 2^(emax - 25).
  5. The exact sum of these terms is cut toward zero to FP32. An exact zero is +0.
Without nonzero products, d = C, and -0 becomes +0. FP32 overflow cannot occur, and a subnormal d
occurs only as such a C.

Storage layouts (Atlas):
  m-register row r: 32 E4M3 codes, element k in byte k
  weight slot row j: output column j's 32 E4M3 codes
"""
import numpy as np

_CODES = np.arange(256)
_FIELD = (_CODES >> 3) & 15
_SIG = np.where(_FIELD > 0, 8 | (_CODES & 7), _CODES & 7)  # |E4M3| = _SIG / 8 x 2^_EXP
_EXP = np.maximum(_FIELD, 1) - 7
_NAN = _CODES & 0x7F == 0x7F
_NEG = _CODES >> 7 == 1


def _cut(value, shift):
    """value x 2^-shift, cut toward zero; shift may be negative."""
    magnitude = np.abs(value)
    magnitude = np.where(shift < 0, magnitude << np.maximum(-shift, 0), magnitude >> np.clip(shift, 0, 63))
    return np.where(value < 0, -magnitude, magnitude)


def dot(a, b, c):
    """d = C + a . b. a and b: (..., 32) E4M3 codes; c: (...) FP32 bits. Returns uint32 FP32 bits."""
    a, b = np.asarray(a, np.uint8), np.asarray(b, np.uint8)
    c = np.asarray(c, np.uint32).astype(np.int64)
    product = _SIG[a] * _SIG[b]  # x 2^(e - 6)
    e = _EXP[a] + _EXP[b]
    product = np.where(_NEG[a] ^ _NEG[b], -product, product)
    c_field = (c >> 23) & 255
    c_exp = np.maximum(c_field, 1) - 127
    c_sig = (c & 0x7F_FFFF) | np.where(c_field > 0, 0x80_0000, 0)  # x 2^(c_exp - 23)
    c_sig = np.where(c >> 31 == 1, -c_sig, c_sig)
    emax = np.maximum(c_exp, np.where(product != 0, e, -1000).max(-1))
    total = (_cut(product, emax[..., None] - e - 19).sum(-1)  # units of 2^(emax - 25)
             + _cut(c_sig, emax - c_exp - 2))
    magnitude = np.abs(total)
    lead = np.frexp(magnitude.astype(np.float64))[1] - 1
    mantissa = np.where(lead > 23, magnitude >> np.maximum(lead - 23, 0), magnitude << np.maximum(23 - lead, 0))
    d = np.where(total < 0, 1 << 31, 0) | (lead + emax - 25 + 127) << 23 | (mantissa & 0x7F_FFFF)
    d = np.where(total == 0, 0, d)
    # Without nonzero products the sum is C exactly; this also covers a subnormal C.
    d = np.where((product == 0).all(-1), np.where(c == 0x8000_0000, 0, c), d)
    d = np.where(c_field == 255, c, d)
    nan = _NAN[a].any(-1) | _NAN[b].any(-1) | ((c_field == 255) & (c & 0x7F_FFFF != 0))
    return np.where(nan, 0x7FFF_FFFF, d).astype(np.uint32)


def matmul(a_rows, w_rows, c):
    """acc (32 x 32 FP32 bits) = c + A @ W^T: A row r is an m-register row, W row j a weight-slot row."""
    a, w = np.asarray(a_rows, np.uint8), np.asarray(w_rows, np.uint8)
    return dot(a[:, None, :], w[None, :, :], c)
