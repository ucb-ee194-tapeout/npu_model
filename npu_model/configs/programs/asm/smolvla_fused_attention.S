seli 0, 127
lui x1, 0x2
li x2, 0x2400
li x3, 0x2800
li x4, 0x2c00
lui x5, 0x3
li x7, 0x3800
lui x9, 0x1
lui x10, 0x2
lui x11, 0x3
lui x12, 0x4
lui x14, 0x6
lui x15, 0x1
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x0, x15, 0
dma.load x2, x9, x15, 1
dma.load x4, x11, x15, 2
vli.all 6, 0x3E00
delay 65
vli.all 8, 0xC2C8
delay 65
vli.all 10, 0
delay 65
vli.all 12, 0
delay 65
vli.all 14, 0
delay 65
dma.wait 0
dma.wait 1
dma.wait 2
dma.load x3, x10, x15, 0
dma.load x5, x12, x15, 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 2, x31, 16
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 3, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 0
delay 32
vmatpop.fp8.mxu0 4, 0, 1
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 2
delay 32
vmatpop.fp8.mxu0 5, 0, 1
delay 32
addi x31, x2, 0  # DMA and VLS bases both count words
vload 16, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 17, x31, 8
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 18, x31, 16
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 19, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 16
delay 32
vmatpop.fp8.mxu0 20, 0, 1
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 18
delay 32
vmatpop.fp8.mxu0 22, 0, 1
delay 32
addi x31, x4, 0  # DMA and VLS bases both count words
vload 24, x31, 0
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 25, x31, 8
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 26, x31, 16
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 27, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 24
delay 32
vmatpop.fp8.mxu0 28, 0, 1
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 26
delay 32
vmatpop.fp8.mxu0 30, 0, 1
delay 32
vtrpose.xlu 20, 20   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 20
delay 32
vmatmul.mxu0 0, 4, 0
delay 96
vtrpose.xlu 22, 22   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 22
delay 32
vmatmul.acc.mxu0 0, 5, 0
delay 96
vmatpop.bf16.mxu0 32, 0
delay 32
vmul.bf16 34, 32, 6
delay 66
vredmax.row.bf16 36, 34
delay 34
vmax.bf16 38, 8, 36
delay 66
vsub.bf16 36, 8, 38
delay 66
vexp 36, 36
delay 66
vmul.bf16 12, 12, 36
delay 66
vmul.bf16 14, 14, 36
delay 66
vsub.bf16 40, 34, 38
delay 66
vexp 40, 40
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 40
delay 32
vmatpop.fp8.mxu0 42, 0, 1
delay 32
vtrpose.xlu 28, 28   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 28
delay 32
vmatmul.mxu1 0, 42, 0
delay 32
vmul.bf16 48, 36, 10
delay 35
vmatpop.bf16.mxu1 52, 0
delay 32
vtrpose.xlu 30, 30   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 30
delay 32
vmatmul.mxu1 0, 42, 0
vredsum.row.bf16 50, 40
delay 35
vmatpop.bf16.mxu1 54, 0
delay 32
vadd.bf16 12, 12, 52
delay 66
vadd.bf16 14, 14, 54
delay 66
vadd.bf16 10, 48, 50
delay 66
vmov 8, 38
delay 66
dma.wait 0
addi x31, x3, 0  # DMA and VLS bases both count words
vload 16, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 17, x31, 8
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 18, x31, 16
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 19, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 16
delay 32
vmatpop.fp8.mxu0 20, 0, 1
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 18
delay 32
vmatpop.fp8.mxu0 22, 0, 1
delay 32
vtrpose.xlu 20, 20   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 20
delay 32
vmatmul.mxu0 0, 4, 0
delay 96
vtrpose.xlu 22, 22   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 22
delay 32
vmatmul.acc.mxu0 0, 5, 0
delay 96
vmatpop.bf16.mxu0 32, 0
delay 32
vmul.bf16 34, 32, 6
delay 66
vredmax.row.bf16 36, 34
delay 34
vmax.bf16 38, 8, 36
delay 66
vsub.bf16 36, 8, 38
delay 66
vexp 36, 36
delay 66
vmul.bf16 12, 12, 36
delay 66
vmul.bf16 14, 14, 36
delay 66
vsub.bf16 40, 34, 38
delay 66
vexp 40, 40
delay 66
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 40
delay 32
vmatpop.fp8.mxu0 42, 0, 1
delay 32
dma.wait 1
addi x31, x5, 0  # DMA and VLS bases both count words
vload 24, x31, 0
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vload 25, x31, 8
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vload 26, x31, 16
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vload 27, x31, 24
delay 34
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 24
delay 32
vmatpop.fp8.mxu0 28, 0, 1
delay 32
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 26
delay 32
vmatpop.fp8.mxu0 30, 0, 1
delay 32
vtrpose.xlu 28, 28   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 28
delay 32
vmatmul.mxu1 0, 42, 0
delay 32
vmul.bf16 48, 36, 10
delay 35
vmatpop.bf16.mxu1 52, 0
delay 32
vtrpose.xlu 30, 30   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 30
delay 32
vmatmul.mxu1 0, 42, 0
vredsum.row.bf16 50, 40
delay 35
vmatpop.bf16.mxu1 54, 0
delay 32
vadd.bf16 12, 12, 52
delay 66
vadd.bf16 14, 14, 54
delay 66
vadd.bf16 10, 48, 50
delay 66
vrecip.bf16 48, 10
delay 66
vmul.bf16 12, 12, 48
delay 66
vmul.bf16 14, 14, 48
delay 66
addi x31, x7, 0  # DMA and VLS bases both count words
vstore 12, x31, 0
delay 34
addi x31, x7, 0  # DMA and VLS bases both count words
vstore 13, x31, 8
delay 34
addi x31, x7, 0  # DMA and VLS bases both count words
vstore 14, x31, 16
delay 34
addi x31, x7, 0  # DMA and VLS bases both count words
vstore 15, x31, 24
delay 34
dma.store x14, x7, x15, 0
dma.wait 0
