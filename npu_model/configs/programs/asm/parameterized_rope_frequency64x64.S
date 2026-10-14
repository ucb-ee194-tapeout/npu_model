li x1, 0x800
lui x2, 0x1
addi x2, x2, -1536
lui x3, 0x1
addi x3, x3, -2048
addi x4, x0, 0
lui x5, 0x2
addi x6, x0, 0
addi x7, x0, 4
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
loop_1:
dma.load x1, x4, x3, 0
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vcos 2, 0
delay 66
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 2, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 3, x31, 8
delay 34
dma.store x5, x2, x3, 0
dma.wait 0
add x4, x4, x3
add x5, x5, x3
addi x6, x6, 1
blt x6, x7, loop_1
addi x0, x0, 0
addi x0, x0, 0
