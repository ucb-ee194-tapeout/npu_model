li x1, 0x800
addi x2, x1, 256
addi x3, x2, 256
lui x6, 0x1
addi x7, x6, 256
addi x8, x0, 0
addi x9, x8, 1024
addi x10, x9, 1024
lui x12, 0x1
addi x13, x12, 1024
addi x14, x0, 1024
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x8, x14, 0
dma.load x2, x9, x14, 1
vli.all 6, 0x3E35
delay 65
dma.wait 0
dma.wait 1
dma.load x3, x10, x14, 0
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 4, x31, 0
delay 34
vtrpose.xlu 2, 2   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 2
delay 32
vmatmul.mxu0 0, 0, 0
delay 96
vmatpop.bf16.mxu0 10, 0
delay 32
vmul.bf16 12, 10, 6
delay 34
vtrpose.xlu 4, 4   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 4
delay 32
vredmax.row.bf16 14, 12
delay 34
vsub.bf16 16, 12, 14
delay 66
vexp 18, 16
delay 66
vredsum.row.bf16 20, 18
delay 39
vrecip.bf16 22, 20
delay 66
vmul.bf16 24, 18, 22
delay 66
seli 0, 127
# Quantize a column-blocked BF16 pair into row-major FP8 via acc1.
vmatpush.acc.bf16.mxu0 1, 24
delay 32
vmatpop.fp8.mxu0 26, 0, 1
delay 32
vmatmul.mxu1 0, 26, 0
delay 35
vmatpop.bf16.mxu1 28, 0
delay 32
addi x31, x6, 0  # DMA and VLS bases both count words
vstore 28, x31, 0
delay 34
addi x31, x7, 0  # DMA and VLS bases both count words
vstore 29, x31, 0
delay 34
dma.store x12, x6, x14, 0
dma.store x13, x7, x14, 1
dma.wait 0
dma.wait 1
