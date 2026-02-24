#include <stdint.h>

/************************* FLOATING POINT TYPES *************************/
typedef union e4m3 {
  struct {
    uint8_t mantissa : 4;
    uint8_t exponent : 3;
    uint8_t sign : 1;
  };

  uint8_t byte : 8;

} e4m3;

typedef union fp16 {
  struct {
    uint16_t mantissa : 11;
    uint16_t exponent : 5;
    uint16_t sign : 1;
  };

  uint16_t half : 16;

} fp16;

typedef union bf16 {
  struct {
    uint16_t mantissa : 8;
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
