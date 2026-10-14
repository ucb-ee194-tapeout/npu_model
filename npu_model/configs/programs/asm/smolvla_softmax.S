li x1, 0x800
li x2, 0xc00
addi x3, x0, 0
addi x4, x3, 1024
lui x5, 0x1
addi x5, x5, -1280
addi x6, x0, 1024
addi x7, x1, 256
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.config x30, 1
dma.wait 1
dma.load x1, x3, x6, 0
dma.load x7, x4, x6, 1
dma.wait 0
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
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
addi x8, x5, 1024
addi x9, x2, 256
dma.store x5, x2, x6, 0
dma.store x8, x9, x6, 1
dma.wait 0
dma.wait 1
