li x1, 0x800
lui x2, 0x1
addi x2, x2, -1536
li x3, 0xc00
lui x4, 0x1
addi x4, x4, -2048
addi x5, x0, 1024
seli 0, 127
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
addi x6, x0, 0
lui x7, 0x1
addi x7, x7, -2048
dma.load x1, x6, x4, 0
dma.load x2, x7, x4, 1
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
dma.wait 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 3, x31, 8
delay 34
vadd.bf16 4, 0, 2
delay 66
vfp8pack 6, 4, 0
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 6, x31, 0
delay 34
lui x8, 0x1
dma.store x8, x3, x5, 0
dma.wait 0
