"""NVFP4 arithmetic of the Atlas NVFP4 MXU (fp4_fedp, from the NVFP4 repo's host/nvfp4_model.py).

One output element is the 64-element NVFP4 dot product of NVIDIA tensor cores (RTX 5090 rule
"rtx5090"): exact 16-element block sums, block terms scaled by both operands' UE4M3 scales, every
term (C included) cut toward zero at 2^(emax - 35), then one cut toward zero to FP32.

Storage layouts (Atlas):
  m-register row r: 64 E2M1 codes, element k in byte k // 2, low nibble for even k
  scale sidecar:    128 bytes, row r at bytes 4r .. 4r + 3, byte g scales elements 16g .. 16g + 15
  FP32 tile in 4 m-registers: register q row r holds columns 8q .. 8q + 7 (little-endian words)
"""
E2M1_MAG2 = [0, 1, 2, 3, 4, 6, 8, 12]  # |E2M1| in units of 0.5


def _pack_rz(value: int, exponent: int) -> int:
    if value == 0:
        return 0
    sign = 0x8000_0000 if value < 0 else 0
    value = abs(value)
    leading = value.bit_length() - 1
    unbiased = leading + exponent
    if unbiased > 127:
        return sign | 0x7F80_0000
    if unbiased < -126:
        shift = -149 - exponent
        mantissa = value >> shift if shift >= 0 else value << -shift
        return sign | mantissa if mantissa else 0
    shift = leading - 23
    mantissa = value >> shift if shift >= 0 else value << -shift
    return sign | (unbiased + 127) << 23 | (mantissa & 0x7F_FFFF)


def _ue4m3(raw: int):
    """UE4M3 scale -> (S, e) with value S / 8 x 2^e; None for NaN (0x7F, bit 7 ignored)."""
    raw &= 127
    if raw == 127:
        return None
    if raw >> 3:
        return 8 + (raw & 7), (raw >> 3) - 7
    return (raw & 7, -6) if raw & 7 else (0, -126)


def dot(a, b, sa, sb, c: int) -> int:
    """d = C + a . b over 64 E2M1 codes with 4 UE4M3 scales per operand; C and d are FP32 bits."""
    terms, nonzero = [], False
    for g in range(4):
        sigma = 0  # sum of 16 products, in units of 1/4
        for x, y in zip(a[16 * g:16 * g + 16], b[16 * g:16 * g + 16]):
            product = E2M1_MAG2[x & 7] * E2M1_MAG2[y & 7]
            nonzero |= product != 0
            sigma += -product if (x ^ y) & 8 else product
        pa, pb = _ue4m3(sa[g]), _ue4m3(sb[g])
        if pa is None or pb is None:
            return 0x7FFF_FFFF
        terms.append((sigma * pa[0] * pb[0], pa[1] + pb[1]))
    c_exp, c_frac = (c >> 23) & 255, c & 0x7F_FFFF
    if c_exp == 255:
        return 0x7FFF_FFFF if c_frac else c
    if not nonzero:
        return 0 if c == 0x8000_0000 else c
    c_sig = c_frac | (0x80_0000 if c_exp else 0)
    c_x = c_exp - 127 if c_exp else -126
    emax = max([c_x] + [x for _, x in terms])
    total = 0
    for t, x in terms:
        aligned = (abs(t) << 27) >> (emax - x)
        total += -aligned if t < 0 else aligned
    aligned = (c_sig << 12) >> (emax - c_x)
    total += -aligned if c >> 31 else aligned
    return _pack_rz(total, emax - 35)


def codes(row_bytes) -> list[int]:
    """64 E2M1 codes of one 32-byte m-register row."""
    out = []
    for byte in row_bytes:
        out += [int(byte) & 15, int(byte) >> 4]
    return out


def f32_to_bf16(x: int) -> int:
    """FP32 bits -> BF16 bits, round to nearest even; NaN -> quiet NaN with the input's sign."""
    if (x >> 23) & 255 == 255 and x & 0x7F_FFFF:
        return (x >> 16) & 0x8000 | 0x7FC0
    return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF


def matmul(a_rows, a_scales, w_rows, w_scales, c):
    """acc (32 x 32 FP32 bits) = c + A @ W^T: A row r and W row j are 32-byte m-register rows,
    scales are 4-byte rows."""
    return [[dot(codes(a_rows[r]), codes(w_rows[j]), a_scales[r], w_scales[j], c[r][j])
             for j in range(len(w_rows))] for r in range(len(a_rows))]
