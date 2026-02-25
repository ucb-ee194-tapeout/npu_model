#include "ee194_softfloat.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define BITMASK_LSB_T(T, N) \
    ((T)((N) == 0           \
             ? 0            \
             : ((N) >= (sizeof(T) * 8) ? (T) ~(T)0 : (((T)1 << (N)) - (T)1))))
/**
 * @brief upcast e4m3 to fp16
 *
 * @param input
 * @return fp16
 */
bf16 e4m3_to_bf16(e4m3 input) {
    bool is_zero = input.exponent == 0 && input.mantissa == 0;
    bool is_nan = input.byte == 0xFF;

    uint16_t fp16_exponent = (input.exponent + 8) & BITMASK_LSB_T(uint16_t, 5);
    uint16_t fp16_mantissa = input.mantissa << 7 & BITMASK_LSB_T(uint16_t, 11) << 7;

    bf16 result;
    if (is_nan) {
        result.half = BITMASK_LSB_T(uint16_t, 15);
        result.sign = input.sign;
    } else if (is_zero) {
        result.half = 0;
        result.sign = input.sign;
    } else {
        result.sign = input.sign;
        result.exponent = fp16_exponent;
        result.mantissa = fp16_mantissa;
    }
    return result;
}

/**
 * @brief upcast e4m3 to fp16
 *
 * @param input
 * @return fp16
 */
fp16 e4m3_to_fp16(e4m3 input) {
    bool is_zero = input.exponent == 0 && input.mantissa == 0;
    bool is_nan = input.byte == 0xFF;

    uint16_t fp16_exponent = (input.exponent + 8) & BITMASK_LSB_T(uint16_t, 5);
    uint16_t fp16_mantissa = input.mantissa << 7;

    fp16 result;
    if (is_nan) {
        result.half = BITMASK_LSB_T(uint16_t, 15);
        result.sign = input.sign;
    } else if (is_zero) {
        result.half = 0;
        result.sign = input.sign;
    } else {
        result.sign = input.sign;
        result.exponent = fp16_exponent;
        result.mantissa = fp16_mantissa;
    }
    return result;
}

/**
 * @brief fp16 * fp16
 *
 * @param a
 * @param b
 * @return fp16
 */
fp16 mul_fp16(fp16 a, fp16 b) {
	
}