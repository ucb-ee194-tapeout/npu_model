addi x1, x0, 0
addi x2, x1, 1024
addi x3, x2, 1024
addi x4, x3, 1024
addi x5, x4, 1024
addi x6, x0, 0
addi x7, x6, 256
addi x8, x7, 256
addi x9, x8, 256
addi x10, x9, 256
addi x11, x10, 256
addi x12, x0, 1024
addi x13, x12, 1024
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x6, x1, x12, 0
dma.load x8, x3, x12, 1
dma.wait 0
dma.wait 1
dma.load x7, x2, x12, 0
dma.load x9, x4, x12, 1
addi x31, x6, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x8, 0  # DMA and VLS bases both count words
vload 1, x31, 0
delay 34
vtrpose.xlu 1, 1   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 1
delay 32
vmatmul.mxu1 0, 0, 0
delay 35
dma.wait 0
dma.wait 1
addi x31, x7, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
addi x31, x9, 0  # DMA and VLS bases both count words
vload 3, x31, 0
delay 34
vtrpose.xlu 3, 3   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 3
delay 32
vmatmul.acc.mxu1 0, 2, 0
delay 35
vmatpop.bf16.mxu1 4, 0
delay 32
addi x31, x10, 0  # DMA and VLS bases both count words
vstore 4, x31, 0
delay 34
addi x31, x11, 0  # DMA and VLS bases both count words
vstore 5, x31, 0
delay 34
dma.store x5, x10, x13, 0
dma.wait 0
