#ifndef EE194_SOFTLOAT_H
#define EE194_SOFTLOAT_H

#include <stdint.h>
#include <stdlib.h>

#define FP16_EXP_WIDTH 5
#define FP16_MANTISSA_WIDTH 10
#define FP16_BIAS -15

#define BF16_EXP_WIDTH 8
#define BF16_MANTISSA_WIDTH 7
#define BF16 -127

#define E4M3_EXP_WIDTH 4
#define E4M3_MANTISSA_WIDTH 3
#define E4M3_BIAS -7

/************************* FLOATING POINT TYPES *************************/
typedef union e4m3 {
  struct {
    uint8_t mantissa : 3;
    uint8_t exponent : 4;
    uint8_t sign : 1;
  };

  uint8_t byte : 8;

} e4m3;

typedef union fp16 {
  struct {
    uint16_t mantissa : 10;
    uint16_t exponent : 5;
    uint16_t sign : 1;
  };

  uint16_t half : 16;

} fp16;

typedef union bf16 {
  struct {
    uint16_t mantissa : 7;
    uint16_t exponent : 8;
    uint16_t sign : 1;
  };

  uint16_t half : 16;

} bf16;

typedef union fp32 {
  struct {
    uint32_t mantissa : 23;
    uint32_t exponent : 8;
    uint32_t sign : 1;
  };

  uint32_t word : 32;
} fp32;

/************************* UPCASTS *************************/
/**
 * @brief upcast e4m3 to bf16
 *
 * @param input
 * @return bf16
 */
bf16 e4m3_to_bf16(e4m3 input);

/**
 * @brief upcast e4m3 to fp16
 *
 * @param input
 * @return fp16
 */
fp16 e4m3_to_fp16(e4m3 input);

/**
 * @brief upcast e4m3 to fp32
 *
 * @param input
 * @return fp32
 */
fp16 e4m3_to_fp32(e4m3 input);

/************************* FP ARITHMETIC *************************/

/**
 * @brief fp16 * fp16
 * 
 * @param a 
 * @param b 
 * @return fp16 
 */
fp16 mul_fp16(fp16 a, fp16 b);

/**
 * @brief generate the anchor exponent from N products and an addend
 * 
 * @param products 
 * @param addend 
 * @return uint8_t 
 */
uint8_t generate_anchor_fp16(fp16 *products, size_t num_products, e4m3 addend);

/**
 * @brief align fp16 to 32-bit fixed-point integer
 * 
 * @param input 
 * @param addend 
 * @return uint32 
 */
uint32_t fp16_to_int_align(fp16 input, uint8_t anchor_exp);

/**
 * @brief align e4m3 to 32-bit fixed-point integer
 *
 * @param input
 * @param addend
 * @return uint32
 */
uint32_t e4m3_to_int_align(e4m3 input, uint8_t anchor_exp);

/**
 * @brief reduce products and addend into a single sum
 * 
 */
int32_t fixed_point_int_reduction(int32_t *products, size_t num_products, int32_t addend);

/**
 * @brief convert 32 bit fixed point integer to bf16
 * 
 * @param value 
 * @return bf16 
 */
bf16 int_to_bf16(int32_t value, uint8_t anchor_exp);

#endif