li x1, 0x800
lui x2, 0x1
addi x2, x2, -1792
lui x3, 0x1
addi x3, x3, -1536
li x4, 0xc00
addi x5, x0, 0
addi x6, x0, 1024
lui x7, 0x1
addi x7, x7, -2048
lui x8, 0x1
addi x9, x0, 1024
lui x10, 0x1
addi x10, x10, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x5, x9, 0
dma.load x2, x6, x9, 1
dma.load x3, x7, x10, 2
dma.wait 0
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0
delay 34
vtrpose.xlu 2, 2   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu1 0, 2
delay 32
vmatmul.mxu1 0, 0, 0
dma.wait 2
addi x31, x3, 0  # DMA and VLS bases both count words
vload 4, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload 5, x31, 8
delay 34
vmatpop.bf16.mxu1 6, 0
delay 32
vadd.bf16 8, 6, 4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 8, x31, 0
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore 9, x31, 8
delay 34
dma.store x8, x4, x10, 0
dma.wait 0
