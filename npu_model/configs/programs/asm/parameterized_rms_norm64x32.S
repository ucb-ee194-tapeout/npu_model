lui x1, 0x2
lui x2, 0x3
lui x3, 0x4
lui x4, 0x5
addi x5, x0, 1024
lui x6, 0x2
addi x6, x6, 1024
lui x7, 0x3
addi x7, x7, 1024
lui x8, 0x4
addi x8, x8, 1024
lui x9, 0x5
addi x9, x9, 1024
addi x10, x0, 0
lui x11, 0x2
addi x11, x11, -2048
addi x12, x0, 0
addi x13, x0, 2
lui x14, 0x1
addi x14, x14, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
lui x15, 0x1
dma.load.ch0 x2, x15, x5
dma.load.ch1 x7, x15, x5
dma.wait.ch0
dma.wait.ch1
lui x15, 0x1
addi x15, x15, 1024
dma.load.ch0 x3, x15, x5
dma.load.ch1 x8, x15, x5
dma.wait.ch0
dma.wait.ch1
vload m8, 0(x3)
delay 34
vload m9, 32(x3)
delay 34
loop_1:
add x15, x10, x5
add x16, x11, x5
dma.load.ch0 x1, x10, x5
dma.load.ch1 x6, x15, x5
vload m6, 0(x2)
delay 34
vload m7, 32(x2)
delay 34
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
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
vstore m6, 0(x4)
delay 34
vstore m7, 32(x4)
delay 34
dma.store.ch0 x11, x4, x5
dma.store.ch1 x16, x9, x5
dma.wait.ch0
dma.wait.ch1
add x10, x10, x14
add x11, x11, x14
addi x12, x12, 1
blt x12, x13, loop_1
addi x0, x0, 0
addi x0, x0, 0