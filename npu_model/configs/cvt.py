"""FP8 and BF16 conversions of the Atlas FP8 path, as the NVIDIA Blackwell conversion instructions
give them (RTX 5090, sm_120a, PTX cvt).

x * G(e) = x * 2^(e - 127) is exact, and each conversion rounds once. Values are bit patterns:
FP32 as uint32 (a BF16 x is FP32 x << 16), BF16 as uint16, E4M3 as uint8. e is 0..255.

  e4m3_rn_satfinite(x, e)  cvt.rn.satfinite.e4m3x2.f32: nearest even, E4M3 subnormals kept,
                           above 448 and +-Inf give +-448 (0x7E, 0xFE), NaN gives 0x7F
  bf16_rne(x, e)           cvt.rn.bf16.f32: nearest even, BF16 subnormals kept, overflow gives
                           +-Inf, NaN gives 0x7FFF
  e4m3_to_f32(code)        cvt.rn.f16x2.e4m3x2 then cvt.f32.f16: exact, NaN gives 0x7FFFFFFF
A result that rounds to zero keeps the sign of x. NaN results have no sign and no payload.
The vectors in baremetal/generators/fp8_silicon (*.cvt.gz) check each rule on the GPU.
"""
import numpy as np

_CODE = np.arange(256)
_FIELD, _M = (_CODE >> 3) & 15, _CODE & 7
_E4M3_F32 = np.where(
    _CODE & 0x7F == 0x7F, 0x7FFF_FFFF,
    (np.where(_CODE >> 7, -1.0, 1.0) * np.where(_FIELD > 0, 8 + _M, _M)
     * 2.0 ** (np.maximum(_FIELD, 1) - 10)).astype(np.float32).view(np.uint32)).astype(np.uint32)


def _round(x, e, mbits, emin):
    """|x| * G(e) rounded to nearest even with mbits fraction bits and emin the smallest normal
    exponent, as (exponent field << mbits) + fraction, without an upper bound; and the sign bit."""
    x = np.asarray(x, np.uint32).astype(np.int64)
    field = (x >> 23) & 255
    sig = (x & 0x7F_FFFF) | np.where(field > 0, 0x80_0000, 0)
    exp = np.maximum(field, 1) - 150 + np.asarray(e, np.int64) - 127  # |x| G(e) = sig x 2^exp
    lead = np.frexp(sig.astype(np.float64))[1] - 1 + exp
    shift = np.clip(np.maximum(lead, emin) - mbits - exp, -mbits, 40)  # quantum 2^(exp + shift)
    up = np.maximum(shift, 0)
    n = (sig >> up) << np.maximum(-shift, 0)
    rest, half = sig & ((1 << up) - 1), (1 << up) >> 1
    n += (shift > 0) & ((rest > half) | ((rest == half) & ((n & 1) == 1)))
    code = np.where(sig == 0, 0, (np.maximum(lead - emin, 0) << mbits) + n)
    return code, x >> 31, field == 255, x & 0x7F_FFFF != 0


def e4m3_rn_satfinite(x, e=127):
    """E4M3 codes of x * G(e); x is FP32 bits."""
    code, sign, special, frac = _round(x, e, 3, -6)
    code = np.where(special, 0x7E, np.minimum(code, 0x7E)) | sign << 7
    return np.where(special & frac, 0x7F, code).astype(np.uint8)


def bf16_rne(x, e=127):
    """BF16 bits of x * G(e); x is FP32 bits."""
    code, sign, special, frac = _round(x, e, 7, -126)
    code = np.where(special, 0x7F80, np.minimum(code, 0x7F80)) | sign << 15
    return np.where(special & frac, 0x7FFF, code).astype(np.uint16)


def e4m3_to_f32(code):
    """FP32 bits of E4M3 codes."""
    return _E4M3_F32[np.asarray(code, np.uint8)]
