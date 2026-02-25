#include <stdio.h>

#include "ee194_softfloat.h"

int main(void) {
    e4m3 fp8_value;
    fp8_value.byte = 0x40;  // 1.0
    fp16 fp16_value = e4m3_to_fp16(fp8_value);

    printf("converting 2.0 in fp8 to fp16:\n");
    printf("result: 0x%2x\n", fp16_value.half);
    return 0;
}