addi x1, x0, 0
addi x2, x1, 1024
addi x3, x2, 1024
addi x4, x3, 1024
addi x5, x4, 1024
addi x6, x0, 0
addi x7, x6, 1024
addi x8, x7, 1024
addi x9, x8, 1024
addi x10, x9, 1024
addi x11, x10, 1024
addi x12, x0, 1024
addi x13, x12, 1024
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x6, x1, x12
dma.load.ch1 x8, x3, x12
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x7, x2, x12
dma.load.ch1 x9, x4, x12
vload m0, 0(x6)
delay 34
vload m1, 0(x8)
delay 34
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
dma.wait.ch0
dma.wait.ch1
vload m2, 0(x7)
delay 34
vload m3, 0(x9)
delay 34
vmatpush.weight.mxu1 w0, m3
delay 32
vmatmul.acc.mxu1 acc0, m2, w0
delay 35
vmatpop.bf16.acc.mxu1 m4, acc0
delay 32
vstore m4, 0(x10)
delay 34
vstore m5, 0(x11)
delay 34
dma.store.ch0 x5, x10, x13
dma.wait.ch0
