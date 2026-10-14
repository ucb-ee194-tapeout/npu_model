li x1, 0x800
li x2, 0xc00
addi x3, x0, 0
lui x4, 0x1
addi x4, x4, -2048
lui x5, 0x1
addi x5, x5, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x1, x3, x5
vli.all m2,  0xBF80
delay 65
vli.all m4,  0x3F80
delay 65
dma.wait.ch0
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vmul.bf16 m6, m0, m2
delay 66
vexp.bf16 m8, m6
delay 66
vadd.bf16 m10, m8, m4
delay 66
vrecip.bf16 m12, m10
delay 66
vmul.bf16 m14, m12, m0
delay 66
addi x31, x2, 0  # DMA and VLS bases both count words
vstore m14, 0(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vstore m15, 8(x31)
delay 34
dma.store.ch0 x4, x2, x5
dma.wait.ch0
