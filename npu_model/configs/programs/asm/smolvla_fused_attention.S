seli e0, 127
lui x1, 0x8
lui x2, 0x9
lui x3, 0xa
lui x4, 0xb
lui x5, 0xc
lui x7, 0xe
lui x9, 0x1
lui x10, 0x2
lui x11, 0x3
lui x12, 0x4
lui x14, 0x6
lui x15, 0x1
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x0, x15
dma.load.ch1 x2, x9, x15
dma.load.ch2 x4, x11, x15
vli.all m6,  0x3E00
delay 65
vli.all m8,  0xC2C8
delay 65
vli.all m10, 0
delay 65
vli.all m12, 0
delay 65
vli.all m14, 0
delay 65
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2
dma.load.ch0 x3, x10, x15
dma.load.ch1 x5, x12, x15
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m0, 0(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m1, 8(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m2, 16(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m3, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m0
delay 32
vmatpop.fp8.acc.mxu0 m4, acc1, e0
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m2
delay 32
vmatpop.fp8.acc.mxu0 m5, acc1, e0
delay 32
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m16, 0(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m17, 8(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m18, 16(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m19, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m20, acc1, e0
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m18
delay 32
vmatpop.fp8.acc.mxu0 m22, acc1, e0
delay 32
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m24, 0(x31)
delay 34
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m25, 8(x31)
delay 34
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m26, 16(x31)
delay 34
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m27, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m24
delay 32
vmatpop.fp8.acc.mxu0 m28, acc1, e0
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m26
delay 32
vmatpop.fp8.acc.mxu0 m30, acc1, e0
delay 32
vtrpose.xlu m20, m20   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vtrpose.xlu m22, m22   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vmul.bf16 m34, m32, m6
delay 66
vredmax.row.bf16 m36, m34
delay 34
vmaximum.bf16 m38, m8, m36
delay 66
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66
vmul.bf16 m12, m12, m36
delay 66
vmul.bf16 m14, m14, m36
delay 66
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m40
delay 32
vmatpop.fp8.acc.mxu0 m42, acc1, e0
delay 32
vtrpose.xlu m28, m28   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m28
delay 32
vmatmul.mxu1 acc0, m42, w0
delay 32
vmul.bf16 m48, m36, m10
delay 35
vmatpop.bf16.acc.mxu1 m52, acc0
delay 32
vtrpose.xlu m30, m30   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m30
delay 32
vmatmul.mxu1 acc0, m42, w0
vredsum.row.bf16 m50, m40
delay 35
vmatpop.bf16.acc.mxu1 m54, acc0
delay 32
vadd.bf16 m12, m12, m52
delay 66
vadd.bf16 m14, m14, m54
delay 66
vadd.bf16 m10, m48, m50
delay 66
vmov m8, m38
delay 66
dma.wait.ch0
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m16, 0(x31)
delay 34
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m17, 8(x31)
delay 34
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m18, 16(x31)
delay 34
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m19, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m20, acc1, e0
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m18
delay 32
vmatpop.fp8.acc.mxu0 m22, acc1, e0
delay 32
vtrpose.xlu m20, m20   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vtrpose.xlu m22, m22   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vmul.bf16 m34, m32, m6
delay 66
vredmax.row.bf16 m36, m34
delay 34
vmaximum.bf16 m38, m8, m36
delay 66
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66
vmul.bf16 m12, m12, m36
delay 66
vmul.bf16 m14, m14, m36
delay 66
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m40
delay 32
vmatpop.fp8.acc.mxu0 m42, acc1, e0
delay 32
dma.wait.ch1
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vload m24, 0(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vload m25, 8(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vload m26, 16(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vload m27, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m24
delay 32
vmatpop.fp8.acc.mxu0 m28, acc1, e0
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m26
delay 32
vmatpop.fp8.acc.mxu0 m30, acc1, e0
delay 32
vtrpose.xlu m28, m28   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m28
delay 32
vmatmul.mxu1 acc0, m42, w0
delay 32
vmul.bf16 m48, m36, m10
delay 35
vmatpop.bf16.acc.mxu1 m52, acc0
delay 32
vtrpose.xlu m30, m30   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m30
delay 32
vmatmul.mxu1 acc0, m42, w0
vredsum.row.bf16 m50, m40
delay 35
vmatpop.bf16.acc.mxu1 m54, acc0
delay 32
vadd.bf16 m12, m12, m52
delay 66
vadd.bf16 m14, m14, m54
delay 66
vadd.bf16 m10, m48, m50
delay 66
vrecip.bf16 m48, m10
delay 66
vmul.bf16 m12, m12, m48
delay 66
vmul.bf16 m14, m14, m48
delay 66
srli x31, x7, 2   # VLS bases count words; DMA bases count bytes
vstore m12, 0(x31)
delay 34
srli x31, x7, 2   # VLS bases count words; DMA bases count bytes
vstore m13, 8(x31)
delay 34
srli x31, x7, 2   # VLS bases count words; DMA bases count bytes
vstore m14, 16(x31)
delay 34
srli x31, x7, 2   # VLS bases count words; DMA bases count bytes
vstore m15, 24(x31)
delay 34
dma.store.ch0 x14, x7, x15
dma.wait.ch0
