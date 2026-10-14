li x1, 0x800
li x4, 0x1400
addi x5, x0, 0
addi x6, x5, 1024
lui x9, 0x1
addi x10, x9, 1024
addi x13, x0, 1024
addi x11, x1, 256
addi x15, x4, 256
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x5, x13, 0
dma.load x11, x6, x13, 1
vli.all 6, 0x3D00
delay 65
vli.all 8, 0x3586
delay 65
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
dma.store x9, x4, x13, 0
dma.store x10, x15, x13, 1
dma.wait 0
dma.wait 1
