li x1, 0x800
lui x2, 0x1
addi x2, x2, -1792
lui x3, 0x1
addi x3, x3, -1536
li x4, 0xc00
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
lui x7, 0x3
addi x7, x7, -1024
lui x8, 0x3
addi x8, x8, 1024
addi x9, x0, 1
addi x10, x0, 1
addi x11, x0, 1024
lui x12, 0x1
addi x12, x12, -2048
lui x13, 0x1
addi x14, x0, 0
lui x20, 0x1
addi x20, x20, -2048
addi x18, x0, 0
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
loop_2:
addi x15, x11, 0
addi x16, x12, 0
addi x17, x13, 0
addi x19, x0, 0
loop_1:
dma.load x1, x14, x5, 0
dma.load x2, x15, x5, 1
dma.wait 0
dma.load x3, x16, x6, 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
dma.wait 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
dma.wait 0
addi x31, x3, 0  # DMA and VLS bases both count words
vload 4, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 5, x31, 8
delay 34
vtrpose.xlu 2, 2   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 2
delay 32
vmatmul.mxu0 0, 0, 0
delay 96
vmatpop.bf16.mxu0 6, 0
delay 32
vadd.bf16 8, 6, 4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 8, x31, 0
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 9, x31, 8
delay 34
dma.store x17, x4, x6, 0
dma.wait 0
addi x15, x15, 1024
add x16, x16, x6
add x17, x17, x6
addi x19, x19, 1
blt x19, x10, loop_1
addi x0, x0, 0
addi x0, x0, 0
addi x14, x14, 1024
add x12, x12, x20
add x13, x13, x20
addi x18, x18, 1
blt x18, x9, loop_2
addi x0, x0, 0
addi x0, x0, 0
