li x1, 0x800
li x2, 0xc00
addi x3, x0, 1024
lui x4, 0x1
addi x4, x4, -1792
lui x5, 0x1
addi x5, x5, -768
addi x6, x0, 0
lui x7, 0x1
addi x8, x0, 0
addi x9, x0, 2
lui x10, 0x1
addi x10, x10, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
loop_1:
add x11, x6, x3
add x12, x7, x3
dma.load x1, x6, x3, 0
dma.load x4, x11, x3, 1
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vredmax.row.bf16 2, 0
delay 34
vsub.bf16 4, 0, 2
delay 66
vexp 6, 4
delay 66
vredsum.row.bf16 8, 6
delay 39
vrecip.bf16 10, 8
delay 66
vmul.bf16 12, 6, 10
delay 66
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 12, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 13, x31, 8
delay 34
dma.store x7, x2, x3, 0
dma.store x12, x5, x3, 1
dma.wait 0
dma.wait 1
add x6, x6, x10
add x7, x7, x10
addi x8, x8, 1
blt x8, x9, loop_1
addi x0, x0, 0
addi x0, x0, 0
