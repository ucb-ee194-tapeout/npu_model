addi x1, x0, 0
addi x2, x0, 1280
lui x3, 0x1
addi x3, x3, -1280
lui x4, 0x2
addi x4, x4, 0
addi x5, x4, 1024
addi x6, x5, 1024
addi x7, x0, 1024
lui x8, 0x1
addi x8, x8, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x4, x1, x7
dma.load.ch1 x5, x2, x7
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x4)
delay 34
vload m1, 0(x5)
delay 34
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
vmatpop.bf16.acc.mxu1 m2, acc0
delay 32
vstore m2, 0(x6)
delay 34
addi x9, x6, 1024
vstore m3, 0(x9)
delay 34
dma.store.ch0 x3, x6, x8
dma.wait.ch0
