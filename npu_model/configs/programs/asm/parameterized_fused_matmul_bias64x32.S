lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x3, x3, -2048
lui x4, 0x3
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
lui x7, 0x3
addi x7, x7, -1024
lui x8, 0x3
addi x8, x8, 1024
addi x9, x0, 2
addi x10, x0, 1
lui x11, 0x1
addi x11, x11, -2048
lui x12, 0x1
addi x12, x12, -1024
lui x13, 0x2
addi x13, x13, -1024
addi x14, x0, 0
lui x20, 0x1
addi x20, x20, -2048
addi x18, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_2:
addi x15, x11, 0
addi x16, x12, 0
addi x17, x13, 0
addi x19, x0, 0
loop_1:
dma.load.ch0 x1, x14, x5
dma.load.ch1 x2, x15, x5
dma.wait.ch0
dma.load.ch0 x3, x16, x6
vload m0, 0(x1)
delay 34
dma.wait.ch1
vload m2, 0(x2)
delay 34
dma.wait.ch0
vload m4, 0(x3)
delay 34
vload m5, 32(x3)
delay 34
vmatpush.weight.mxu0 w0, m2
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m6, acc0
delay 32
vadd.bf16 m8, m6, m4
delay 66
vstore m8, 0(x4)
delay 34
vstore m9, 32(x4)
delay 34
dma.store.ch0 x17, x4, x6
dma.wait.ch0
addi x15, x15, 1024
add x16, x16, x6
add x17, x17, x6
addi x19, x19, 1
blt x19, x10, loop_1
addi x0, x0, 0
addi x0, x0, 0
addi x14, x14, 1024
add x12, x12, x20
add x13, x13, x20
addi x18, x18, 1
blt x18, x9, loop_2
addi x0, x0, 0
addi x0, x0, 0