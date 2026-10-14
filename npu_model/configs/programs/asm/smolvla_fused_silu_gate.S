li x1, 0x800
li x2, 0xc00
addi x3, x0, 0
lui x4, 0x1
addi x4, x4, -2048
lui x5, 0x1
addi x5, x5, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x3, x5, 0
vli.all 2, 0xBF80
delay 65
vli.all 4, 0x3F80
delay 65
dma.wait 0
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vmul.bf16 6, 0, 2
delay 66
vexp 8, 6
delay 66
vadd.bf16 10, 8, 4
delay 66
vrecip.bf16 12, 10
delay 66
vmul.bf16 14, 12, 0
delay 66
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 14, x31, 0
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vstore 15, x31, 8
delay 34
dma.store x4, x2, x5, 0
dma.wait 0
