lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x3, x3, -2048
lui x4, 0x3
addi x4, x4, -1024
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
addi x7, x0, 1024
addi x8, x0, 1024
lui x9, 0x1
addi x9, x9, -2048
addi x10, x0, 1
addi x11, x0, 1
addi x23, x0, 2
addi x25, x0, 1024
addi x26, x0, 1024
lui x27, 0x1
addi x27, x27, -2048
addi x28, x0, 0
lui x29, 0x1
addi x29, x29, -2048
lui x30, 0x1
addi x24, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_3:
addi x15, x28, 0
addi x14, x30, 0
addi x13, x29, 0
addi x20, x0, 0
loop_2:
addi x16, x13, 0
addi x17, x14, 0
addi x21, x0, 0
loop_1:
addi x18, x15, 0
addi x19, x16, 0
dma.load.ch0 x1, x18, x5
dma.load.ch1 x2, x19, x5
dma.wait.ch0
vload m0, 0(x1)
delay 34
dma.wait.ch1
vload m1, 0(x2)
delay 34
vmatpush.weight.mxu0 w0, m1
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
addi x18, x18, 1024
add x19, x19, x7
vmatpop.bf16.acc.mxu0 m2, acc0
delay 32
vstore m2, 0(x3)
delay 34
vstore m3, 0(x4)
delay 34
dma.store.ch0 x17, x3, x6
dma.wait.ch0
addi x16, x16, 1024
add x17, x17, x6
addi x21, x21, 1
blt x21, x11, loop_1
addi x0, x0, 0
addi x0, x0, 0
add x15, x15, x8
add x14, x14, x9
addi x20, x20, 1
blt x20, x10, loop_2
addi x0, x0, 0
addi x0, x0, 0
add x28, x28, x25
add x29, x29, x26
add x30, x30, x27
addi x24, x24, 1
blt x24, x23, loop_3
addi x0, x0, 0
addi x0, x0, 0