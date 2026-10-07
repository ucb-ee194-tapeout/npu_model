seli e0, 127
addi x1, x0, 0
lui x2, 0x1
lui x3, 0x2
lui x4, 0x3
lui x5, 0x3
addi x5, x5, 1024
addi x6, x0, 0
lui x7, 0x9
lui x8, 0x9
addi x8, x8, 1024
lui x9, 0x1
addi x10, x0, 1024
lui x13, 0x1
lui x14, 0x5
addi x15, x0, 1
addi x17, x0, 4
addi x16, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x4, x7, x10
dma.wait.ch0
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m2, 0(x31)
delay 34
srli x31, x4, 2   # VLS bases count words; DMA bases count bytes
vload m3, 0(x31)
delay 34
loop_2:
dma.load.ch0 x1, x6, x9
dma.wait.ch0
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m12, 0(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m13, 8(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m12
delay 32
vmatpop.fp8.acc.mxu0 m0, acc1, e0
delay 32
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m12, 16(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m13, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m12
delay 32
vmatpop.fp8.acc.mxu0 m1, acc1, e0
delay 32
vli.all m4,  0xC2C8
delay 65
vli.all m6, 0
delay 65
vli.all m8, 0
delay 65
vli.all m10, 0
delay 65
addi x11, x13, 0
addi x12, x14, 0
addi x18, x0, 0
loop_1:
dma.load.ch0 x2, x11, x9
dma.load.ch1 x3, x12, x9
dma.wait.ch0
dma.wait.ch1
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m12, 0(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m13, 8(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m12
delay 32
vmatpop.fp8.acc.mxu0 m14, acc1, e0
delay 32
vtrpose.xlu m14, m14   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m14
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m12, 16(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m13, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m12
delay 32
vmatpop.fp8.acc.mxu0 m14, acc1, e0
delay 32
vtrpose.xlu m14, m14   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m14
delay 32
vmatmul.acc.mxu0 acc0, m1, w0
delay 96
vmatpop.bf16.acc.mxu0 m20, acc0
delay 32
vmul.bf16 m22, m20, m2
delay 66
vredmax.row.bf16 m24, m22
delay 34
vmaximum.bf16 m26, m4, m24
delay 66
vsub.bf16 m24, m4, m26
delay 66
vexp.bf16 m24, m24
delay 66
vmul.bf16 m8, m8, m24
delay 66
vmul.bf16 m10, m10, m24
delay 66
vsub.bf16 m28, m22, m26
delay 66
vexp.bf16 m28, m28
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m28
delay 32
vmatpop.fp8.acc.mxu0 m30, acc1, e0
delay 32
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m16, 0(x31)
delay 34
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m17, 8(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m18, acc1, e0
delay 32
vtrpose.xlu m18, m18   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m18
delay 32
vmatmul.mxu0 acc0, m30, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vadd.bf16 m8, m8, m32
delay 66
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m16, 16(x31)
delay 34
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vload m17, 24(x31)
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m18, acc1, e0
delay 32
vtrpose.xlu m18, m18   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m18
delay 32
vmatmul.mxu0 acc0, m30, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vadd.bf16 m10, m10, m32
delay 66
vmul.bf16 m34, m24, m6
delay 66
vredsum.row.bf16 m36, m28
delay 39
vadd.bf16 m6, m34, m36
delay 66
vmov m4, m26
delay 66
add x11, x11, x9
add x12, x12, x9
addi x18, x18, 1
blt x18, x17, loop_1
addi x0, x0, 0
addi x0, x0, 0
vrecip.bf16 m34, m6
delay 66
vmul.bf16 m8, m8, m34
delay 66
vmul.bf16 m10, m10, m34
delay 66
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vstore m8, 0(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vstore m9, 8(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vstore m10, 16(x31)
delay 34
srli x31, x5, 2   # VLS bases count words; DMA bases count bytes
vstore m11, 24(x31)
delay 34
dma.store.ch0 x8, x5, x9
dma.wait.ch0
add x6, x6, x9
add x8, x8, x9
addi x16, x16, 1
blt x16, x15, loop_2
addi x0, x0, 0
addi x0, x0, 0
