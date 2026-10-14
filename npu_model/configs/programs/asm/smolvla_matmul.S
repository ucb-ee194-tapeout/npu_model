addi x1, x0, 0
addi x2, x0, 1280
lui x3, 0x1
addi x3, x3, -1280
lui x4, 0x1
addi x4, x4, -2048
addi x5, x4, 256
addi x6, x5, 256
addi x7, x0, 1024
lui x8, 0x1
addi x8, x8, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x4, x1, x7, 0
dma.load x5, x2, x7, 1
dma.wait 0
dma.wait 1
addi x31, x4, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vload 1, x31, 0
delay 34
vtrpose.xlu 1, 1   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 1
delay 32
vmatmul.mxu1 0, 0, 0
delay 35
vmatpop.bf16.mxu1 2, 0
delay 32
addi x31, x6, 0  # DMA and VLS bases both count words
vstore 2, x31, 0
delay 34
addi x9, x6, 256
addi x31, x9, 0  # DMA and VLS bases both count words
vstore 3, x31, 0
delay 34
dma.store x3, x6, x8, 0
dma.wait 0
