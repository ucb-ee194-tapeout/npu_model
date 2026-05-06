lui x1, 0x2
lui x2, 0x3
addi x3, x0, 1024
lui x4, 0x2
addi x4, x4, 1024
addi x5, x0, 0
lui x6, 0x1
addi x7, x0, 0
addi x8, x0, 2
lui x9, 0x1
addi x9, x9, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_1:
add x10, x5, x3
dma.load.ch0 x1, x5, x3
dma.load.ch1 x4, x10, x3
dma.wait.ch0
vload m0, 0(x1)
delay 34
dma.wait.ch1
vload m1, 32(x1)
delay 34
vredsum.row.bf16 m4, m0
delay 39
vstore m4, 0(x2)
delay 34
dma.store.ch0 x6, x2, x3
dma.wait.ch0
add x5, x5, x9
add x6, x6, x3
addi x7, x7, 1
blt x7, x8, loop_1
addi x0, x0, 0
addi x0, x0, 0