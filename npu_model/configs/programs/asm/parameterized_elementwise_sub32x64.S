lui x1, 0x2
lui x2, 0x3
addi x2, x2, -2048
lui x3, 0x3
lui x4, 0x1
addi x4, x4, -2048
addi x5, x0, 0
lui x6, 0x1
lui x7, 0x2
addi x8, x0, 0
addi x9, x0, 2
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_1:
dma.load.ch0 x1, x5, x4
dma.load.ch1 x2, x6, x4
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
dma.wait.ch1
vload m2, 0(x2)
delay 34
vload m3, 32(x2)
delay 34
vsub.bf16 m4, m0, m2
delay 66
vstore m4, 0(x3)
delay 34
vstore m5, 32(x3)
delay 34
dma.store.ch0 x7, x3, x4
dma.wait.ch0
add x5, x5, x4
add x6, x6, x4
add x7, x7, x4
addi x8, x8, 1
blt x8, x9, loop_1
addi x0, x0, 0
addi x0, x0, 0