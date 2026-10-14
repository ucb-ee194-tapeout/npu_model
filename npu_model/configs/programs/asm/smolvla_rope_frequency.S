lui x1, 0x1
addi x1, x1, -2048
addi x4, x0, 0
lui x5, 0x1
addi x5, x5, -1280
lui x6, 0x1
addi x6, x6, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x4, x6, 0
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vcos 2, 0
delay 66
addi x31, x1, 0  # DMA and VLS bases both count words
vstore 2, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vstore 3, x31, 8
delay 34
dma.store x5, x1, x6, 0
dma.wait 0
