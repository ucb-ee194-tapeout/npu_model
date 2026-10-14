li x1, 0x800
lui x2, 0x1
addi x2, x2, -1536
li x3, 0xc00
lui x4, 0x1
addi x4, x4, -2048
addi x5, x0, 0
lui x6, 0x1
addi x6, x6, -2048
lui x7, 0x1
addi x8, x0, 0
addi x9, x0, 1
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
loop_1:
dma.load x1, x5, x4, 0
dma.load x2, x6, x4, 1
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
vmul.bf16 4, 0, 2
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 4, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 5, x31, 8
delay 34
dma.store x7, x3, x4, 0
dma.wait 0
add x5, x5, x4
add x6, x6, x4
add x7, x7, x4
addi x8, x8, 1
blt x8, x9, loop_1
addi x0, x0, 0
addi x0, x0, 0
