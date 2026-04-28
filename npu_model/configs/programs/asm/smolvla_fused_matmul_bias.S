lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x3, x3, -2048
lui x4, 0x3
addi x5, x0, 0
addi x6, x0, 1024
lui x7, 0x1
addi x7, x7, -2048
lui x8, 0x1
addi x9, x0, 1024
lui x10, 0x1
addi x10, x10, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x5, x9
dma.load.ch1 x2, x6, x9
dma.load.ch2 x3, x7, x10
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m2, 0(x2)
delay 34
vmatpush.weight.mxu1 w0, m2
delay 32
vmatmul.mxu1 acc0, m0, w0
dma.wait.ch2
vload m4, 0(x3)
delay 34
vload m5, 32(x3)
delay 34
vmatpop.bf16.acc.mxu1 m6, acc0
delay 32
vadd.bf16 m8, m6, m4
delay 66
vstore m8, 0(x4)
delay 34
vstore m9, 32(x4)
delay 34
dma.store.ch0 x8, x4, x10
dma.wait.ch0
