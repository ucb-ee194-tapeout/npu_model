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
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x1, x5, x13
dma.load.ch1 x11, x6, x13
vli.all m6, 0x3D00
delay 65
vli.all m8, 0x3586
delay 65
dma.wait.ch0
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vsquare.bf16 m2, m0
delay 66
vredsum.row.bf16 m4, m2
delay 39
vmul.bf16 m10, m4, m6
delay 66
vadd.bf16 m12, m10, m8
delay 66
vsqrt.bf16 m14, m12
delay 66
vrecip.bf16 m4, m14
delay 66
vmul.bf16 m6, m0, m4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m6, 0(x31)
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m7, 8(x31)
delay 34
dma.store.ch0 x9, x4, x13
dma.store.ch1 x10, x15, x13
dma.wait.ch0
dma.wait.ch1
