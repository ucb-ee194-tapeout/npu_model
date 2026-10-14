addi x1, x1, 0
addi x2, x2, 8
loop:
    addi x1, x1, 1
    blt x1, x2, loop
    delay 32
vmatmul.mxu1 1, 1, 1
addi x4, x4, 1
addi x5, x5, 1
delay 0
