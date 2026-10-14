li x1, 0x800
li x5, 0x1800
addi x6, x0, 0
addi x7, x6, 1024
lui x11, 0x1
addi x11, x11, 1024
addi x12, x11, 1024
addi x13, x0, 1024
addi x14, x1, 256
addi x18, x5, 256
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x6, x13, 0
dma.load x14, x7, x13, 1
vli.all 2, 0x3D37
delay 65
vli.all 4, 0x3F4C
delay 65
vli.all 6, 0x3F00
delay 65
vli.all 8, 0x3F80
delay 65
dma.wait 0
dma.wait 1
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8
delay 34
vmul.bf16 10, 0, 0
delay 66
vmul.bf16 12, 10, 0
delay 66
vmul.bf16 14, 12, 2
delay 66
vadd.bf16 16, 0, 14
delay 66
vmul.bf16 18, 16, 4
delay 66
vtanh 20, 18
delay 66
vadd.bf16 22, 8, 20
delay 66
vmul.bf16 24, 0, 6
delay 66
vmul.bf16 26, 24, 22
delay 66
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 26, x31, 0
delay 34
dma.store x11, x5, x13, 0
addi x31, x18, 0  # DMA and VLS bases both count words
vstore 27, x31, 0
delay 34
dma.store x12, x18, x13, 1
dma.wait 0
dma.wait 1
