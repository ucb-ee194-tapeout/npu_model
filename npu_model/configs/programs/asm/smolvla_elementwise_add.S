lui x1, 0x1
li x2, 0x1400
li x3, 0x1800
addi x4, x0, 0
lui x5, 0x1
lui x6, 0x2
lui x7, 0x1
addi x7, x7, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.load x1, x4, x7, 0
dma.load x2, x5, x7, 1
dma.wait 0
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 3, x31, 8
delay 34
vadd.bf16 4, 0, 2
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 4, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 5, x31, 8
delay 34
dma.store x6, x3, x7, 0
dma.wait 0
