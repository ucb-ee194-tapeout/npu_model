li x1, 0x800
lui x2, 0x1
addi x2, x2, -1792
lui x3, 0x1
addi x3, x3, -1536
lui x4, 0x3
addi x4, x4, -1024
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
addi x7, x0, 1024
addi x8, x0, 1024
lui x9, 0x1
addi x9, x9, -2048
addi x10, x0, 1
addi x11, x0, 1
addi x13, x0, 1024
lui x14, 0x1
addi x14, x14, -2048
addi x15, x0, 0
addi x20, x0, 0
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.wait 0
dma.wait 1
loop_2:
addi x16, x13, 0
addi x17, x14, 0
addi x21, x0, 0
loop_1:
addi x18, x15, 0
addi x19, x16, 0
dma.load x1, x18, x5, 0
dma.load x2, x19, x5, 1
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
dma.wait 1
addi x31, x2, 0  # DMA and VLS bases both count words
vload 1, x31, 0
delay 34
vtrpose.xlu 1, 1   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 1
delay 32
vmatmul.mxu1 0, 0, 0
delay 35
addi x18, x18, 1024
add x19, x19, x7
vmatpop.bf16.mxu1 2, 0
delay 32
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 2, x31, 0
delay 34
srli x31, x4, 2   # DMA and VLS bases both count words
vstore 3, x31, 0
delay 34
dma.store x17, x3, x6, 0
dma.wait 0
addi x16, x16, 1024
add x17, x17, x6
addi x21, x21, 1
blt x21, x11, loop_1
addi x0, x0, 0
addi x0, x0, 0
add x15, x15, x8
add x14, x14, x9
addi x20, x20, 1
blt x20, x10, loop_2
addi x0, x0, 0
addi x0, x0, 0
