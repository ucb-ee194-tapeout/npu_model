lui x1, 0x2
lui x2, 0x3
addi x3, x0, 1024
lui x4, 0x2
addi x4, x4, 1024
lui x5, 0x3
addi x5, x5, 1024
addi x6, x0, 0
lui x7, 0x2
addi x7, x7, -2048
addi x8, x0, 0
addi x9, x0, 3
lui x10, 0x1
addi x10, x10, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_1:
add x11, x6, x3
add x12, x7, x3
dma.load.ch0 x1, x6, x3
dma.load.ch1 x4, x11, x3
dma.wait.ch0
vload m0, 0(x1)
delay 34
dma.wait.ch1
vload m1, 32(x1)
delay 34
vredmax.row.bf16 m2, m0
delay 34
vsub.bf16 m4, m0, m2
delay 66
vexp.bf16 m6, m4
delay 66
vredsum.row.bf16 m8, m6
delay 39
vrecip.bf16 m10, m8
delay 66
vmul.bf16 m12, m6, m10
delay 66
vstore m12, 0(x2)
delay 34
vstore m13, 32(x2)
delay 34
dma.store.ch0 x7, x2, x3
dma.store.ch1 x12, x5, x3
dma.wait.ch0
dma.wait.ch1
add x6, x6, x10
add x7, x7, x10
addi x8, x8, 1
blt x8, x9, loop_1
addi x0, x0, 0
addi x0, x0, 0