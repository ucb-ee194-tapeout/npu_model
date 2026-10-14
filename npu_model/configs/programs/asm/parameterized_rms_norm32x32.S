li x1, 0x800
li x2, 0xc00
lui x3, 0x1
li x4, 0x1400
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -1792
lui x7, 0x1
addi x7, x7, -768
lui x8, 0x1
addi x8, x8, 256
lui x9, 0x1
addi x9, x9, 1280
addi x10, x0, 0
lui x11, 0x1
addi x12, x0, 0
addi x13, x0, 1
lui x14, 0x1
addi x14, x14, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
lui x15, 0x1
addi x15, x15, -2048
dma.load x2, x15, x5, 0
dma.load x7, x15, x5, 1
dma.wait 0
dma.wait 1
lui x15, 0x1
addi x15, x15, -1024
dma.load x3, x15, x5, 0
dma.load x8, x15, x5, 1
dma.wait 0
dma.wait 1
addi x31, x3, 0  # DMA and VLS bases both count words
vload 8, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 9, x31, 8
delay 34
loop_1:
add x15, x10, x5
add x16, x11, x5
dma.load x1, x10, x5, 0
dma.load x6, x15, x5, 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 6, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 7, x31, 8
delay 34
dma.wait 0
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vsquare.bf16 2, 0
delay 66
vredsum.row.bf16 4, 2
delay 39
vmul.bf16 10, 4, 6
delay 66
vadd.bf16 12, 10, 8
delay 66
vsqrt 14, 12
delay 66
vrecip.bf16 4, 14
delay 66
vmul.bf16 6, 0, 4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 6, x31, 0
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 7, x31, 8
delay 34
dma.store x11, x4, x5, 0
dma.store x16, x9, x5, 1
dma.wait 0
dma.wait 1
add x10, x10, x14
add x11, x11, x14
addi x12, x12, 1
blt x12, x13, loop_1
addi x0, x0, 0
addi x0, x0, 0
