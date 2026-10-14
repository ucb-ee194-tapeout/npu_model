li x1, 0x800
lui x2, 0x1
addi x2, x2, -1792
li x3, 0xc00
addi x4, x0, 1024
addi x5, x0, 0
lui x6, 0x1
addi x7, x0, 0
addi x8, x0, 2
lui x9, 0x1
addi x9, x9, -2048
seli 0, 127
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
loop_1:
add x10, x5, x4
dma.load x1, x5, x4, 0
dma.load x2, x10, x4, 1
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
dma.wait 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 1, x31, 0
delay 34
vfp8pack 2, 0, 0
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 2, x31, 0
delay 34
dma.store x6, x3, x4, 0
dma.wait 0
add x5, x5, x9
add x6, x6, x4
addi x7, x7, 1
blt x7, x8, loop_1
addi x0, x0, 0
addi x0, x0, 0
