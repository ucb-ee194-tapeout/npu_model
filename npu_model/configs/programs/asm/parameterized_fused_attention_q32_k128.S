seli 0, 127
addi x1, x0, 0
addi x2, x0, 1024
li x3, 0x800
li x4, 0xc00
lui x5, 0x1
addi x5, x5, -768
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
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
dma.load x4, x7, x10, 0
dma.wait 0
addi x31, x4, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 3, x31, 0
delay 34
loop_2:
dma.load x1, x6, x9, 0
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 12, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 13, x31, 8
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 12
delay 32
vmatpop.fp8.mxu0 0, 0, 1
delay 32
addi x31, x1, 0  # DMA and VLS bases both count words
vload 12, x31, 16
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 13, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 12
delay 32
vmatpop.fp8.mxu0 1, 0, 1
delay 32
vli.all 4, 0xC2C8
delay 65
vli.all 6, 0
delay 65
vli.all 8, 0
delay 65
vli.all 10, 0
delay 65
addi x11, x13, 0
addi x12, x14, 0
addi x18, x0, 0
loop_1:
dma.load x2, x11, x9, 0
dma.load x3, x12, x9, 1
dma.wait 0
dma.wait 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 12, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 13, x31, 8
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 12
delay 32
vmatpop.fp8.mxu0 14, 0, 1
delay 32
vtrpose.xlu 14, 14   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 14
delay 32
vmatmul.mxu0 0, 0, 0
delay 96
addi x31, x2, 0  # DMA and VLS bases both count words
vload 12, x31, 16
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 13, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 12
delay 32
vmatpop.fp8.mxu0 14, 0, 1
delay 32
vtrpose.xlu 14, 14   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 14
delay 32
vmatmul.acc.mxu0 0, 1, 0
delay 96
vmatpop.bf16.mxu0 20, 0
delay 32
vmul.bf16 22, 20, 2
delay 66
vredmax.row.bf16 24, 22
delay 34
vmax.bf16 26, 4, 24
delay 66
vsub.bf16 24, 4, 26
delay 66
vexp 24, 24
delay 66
vmul.bf16 8, 8, 24
delay 66
vmul.bf16 10, 10, 24
delay 66
vsub.bf16 28, 22, 26
delay 66
vexp 28, 28
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 28
delay 32
vmatpop.fp8.mxu0 30, 0, 1
delay 32
addi x31, x3, 0  # DMA and VLS bases both count words
vload 16, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 17, x31, 8
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 16
delay 32
vmatpop.fp8.mxu0 18, 0, 1
delay 32
vtrpose.xlu 18, 18   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 18
delay 32
vmatmul.mxu0 0, 30, 0
delay 96
vmatpop.bf16.mxu0 32, 0
delay 32
vadd.bf16 8, 8, 32
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vload 16, x31, 16
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 17, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 16
delay 32
vmatpop.fp8.mxu0 18, 0, 1
delay 32
vtrpose.xlu 18, 18   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 18
delay 32
vmatmul.mxu0 0, 30, 0
delay 96
vmatpop.bf16.mxu0 32, 0
delay 32
vadd.bf16 10, 10, 32
delay 66
vmul.bf16 34, 24, 6
delay 66
vredsum.row.bf16 36, 28
delay 39
vadd.bf16 6, 34, 36
delay 66
vmov 4, 26
delay 66
add x11, x11, x9
add x12, x12, x9
addi x18, x18, 1
blt x18, x17, loop_1
addi x0, x0, 0
addi x0, x0, 0
vrecip.bf16 34, 6
delay 66
vmul.bf16 8, 8, 34
delay 66
vmul.bf16 10, 10, 34
delay 66
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 8, x31, 0
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 9, x31, 8
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 10, x31, 16
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 11, x31, 24
delay 34
dma.store x8, x5, x9, 0
dma.wait 0
add x6, x6, x9
add x8, x8, x9
addi x16, x16, 1
blt x16, x15, loop_2
addi x0, x0, 0
addi x0, x0, 0
