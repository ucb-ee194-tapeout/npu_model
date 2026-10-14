li x1, 0x800
lui x2, 0x1
addi x2, x2, -1536
lui x3, 0x1
addi x3, x3, -2048
addi x4, x0, 0
lui x5, 0x1
addi x6, x0, 0
addi x7, x0, 2
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
vli.all m2,  0xBF80
delay 65
vli.all m4,  0x3F80
delay 65
loop_1:
dma.load.ch0 x1, x4, x3
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
dma.store.ch0 x5, x2, x3
dma.wait.ch0
add x4, x4, x3
add x5, x5, x3
addi x6, x6, 1
blt x6, x7, loop_1
addi x0, x0, 0
addi x0, x0, 0
