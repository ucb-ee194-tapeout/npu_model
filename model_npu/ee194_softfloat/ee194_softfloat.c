#include "ee194_softfloat.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

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
bf16 e4m3_to_bf16(e4m3 input)
{
	// return NULL;
}

/**
 * @brief upcast e4m3 to fp16
 *
 * @param input
 * @return fp16
 */
fp16 e4m3_to_fp16(e4m3 input)
{
	bool is_zero = input.exponent == 0 && input.mantissa == 0;
	bool is_nan = input.byte == 0xFF;

	uint16_t fp16_exponent = (input.exponent + 8) & BITMASK_LSB_T(uint16_t, 5);
	uint16_t fp16_mantissa = input.mantissa << 7;

	fp16 result;
	if (is_nan)
	{
		result.half = BITMASK_LSB_T(uint16_t, 15);
		result.sign = input.sign;
	}
	else if (is_zero)
	{
		result.half = 0;
		result.sign = input.sign;
	}
	else
	{
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
fp16 mul_fp16(fp16 a, fp16 b)
{
	fp16 result;

	if (a.exponent == 0 || b.exponent == 0)
	{
		result.half = 0;
		result.sign = a.sign ^ b.sign;
		return result;
	}

	result.sign = a.sign ^ b.sign;

	int exp_a = a.exponent - 15;
	int exp_b = b.exponent - 15;

	int exp = exp_a + exp_b;

	uint32_t sig_a = (1u << 10) | a.mantissa;
	uint32_t sig_b = (1u << 10) | b.mantissa;

	uint32_t product = sig_a * sig_b;

	if (product & (1u << 21))
	{
		product >>= 1;
		exp += 1;
	}

	uint32_t sig = product >> 10;
	uint32_t remainder = product & 0x3FF;

	uint32_t halfway = 1u << 9;

	if (remainder > halfway || (remainder == halfway && (sig & 1)))
	{
		sig += 1;
		if (sig & (1u << 11))
		{
			sig >>= 1;
			exp += 1;
		}
	}

	uint16_t mantissa = sig & 0x3FF;

	int final_exp = exp + 15;

	if (final_exp >= 31)
	{
		result.exponent = 31;
		result.mantissa = 0;
		return result;
	}

	if (final_exp <= 0)
	{
		result.half = 0;
		result.sign = result.sign;
		return result;
	}

	result.exponent = final_exp;
	result.mantissa = mantissa;

	return result;
}

/**
 * @brief generate the anchor exponent from N products and an addend
 *
 * @param products
 * @param addend
 * @return uint8_t
 */
uint8_t generate_anchor_fp16(fp16 *products, size_t num_products, e4m3 addend)
{
	uint8_t max_fp16_exponent;

	for (int i = 0; i < num_products; i++)
	{
		if (max_fp16_exponent < (products[i].exponent + FP16_BIAS))
		{
			max_fp16_exponent = (products[i].exponent + FP16_BIAS);
		}
	}

	uint8_t max_exponent = max_fp16_exponent > (addend.exponent + E4M3_BIAS) ? max_fp16_exponent : (addend.exponent + E4M3_BIAS);
	uint8_t anchor_headroom = ceil(log2(32 + 1)) + 1;

	return max_exponent + anchor_headroom;
}

/**
 * @brief align fp16 to 32-bit fixed-point integer
 *
 * @param input
 * @param addend
 * @return uint32
 */
uint32_t fp16_to_int_align(fp16 input, uint8_t anchor_exp)
{
	const int int_width = 32;
	const int frac_bits = 10; // sigWidth - 1 for fp16
	const int exp_bias = 15;

	if (input.exponent == 0 && input.mantissa == 0)
	{
		return 0;
	}
	uint32_t full_sig;
	int32_t unbiased_exp;

	if (input.exponent == 0)
	{
		full_sig = input.mantissa;
		unbiased_exp = 1 - exp_bias;
	}
	else
	{
		// normalized
		full_sig = (1u << frac_bits) | input.mantissa;
		unbiased_exp = (int32_t)input.exponent - exp_bias;
	}

	// shiftRight = frac_bits + anchorExp - (int_width - 1) - unbExp
	int32_t shiftRight =
		frac_bits +
		(int32_t)anchor_exp -
		(int_width - 1) -
		unbiased_exp;

	uint32_t magnitude = 0;

	if (shiftRight >= int_width)
	{
		magnitude = 0;
	}
	else if (shiftRight >= 0)
	{
		magnitude = full_sig >> shiftRight;
	}
	else if (shiftRight >= -(int_width - 1))
	{
		magnitude = full_sig << (-shiftRight);
	}
	else
	{
		magnitude = 0;
	}

	int32_t signed_val = input.sign ? -(int32_t)magnitude : (int32_t)magnitude;

	return (uint32_t)signed_val;
}

/**
 * @brief align e4m3 to 32-bit fixed-point integer
 *
 * @param input
 * @param addend
 * @return uint32
 */
uint32_t e4m3_to_int_align(e4m3 input, uint8_t anchor_exp)
{
	const int int_width = 32;
	const int frac_bits = 3;
	const int exp_bias = 7;

	if (input.exponent == 0 && input.mantissa == 0)
	{
		return 0;
	}
	uint32_t full_sig;
	int32_t unbiased_exp;

	if (input.exponent == 0)
	{
		full_sig = input.mantissa;
		unbiased_exp = 1 - exp_bias;
	}
	else
	{
		// normalized
		full_sig = (1u << frac_bits) | input.mantissa;
		unbiased_exp = (int32_t)input.exponent - exp_bias;
	}

	// shiftRight = frac_bits + anchorExp - (int_width - 1) - unbExp
	int32_t shiftRight =
		frac_bits +
		(int32_t)anchor_exp -
		(int_width - 1) -
		unbiased_exp;

	uint32_t magnitude = 0;

	if (shiftRight >= int_width)
	{
		magnitude = 0;
	}
	else if (shiftRight >= 0)
	{
		magnitude = full_sig >> shiftRight;
	}
	else if (shiftRight >= -(int_width - 1))
	{
		magnitude = full_sig << (-shiftRight);
	}
	else
	{
		magnitude = 0;
	}

	int32_t signed_val = input.sign ? -(int32_t)magnitude : (int32_t)magnitude;

	return (uint32_t)signed_val;
}

/**
 * @brief reduce products and addend into a single sum
 *
 */
uint32_t fixed_point_int_reduction(uint32_t *products, size_t num_products, uint32_t addend)
{
	uint32_t reduction;
	for (int i = 0; i < num_products; i++)
	{
		reduction += products[i];
	}

	return reduction + addend;
}

/**
 * @brief convert 32 bit fixed point integer to bf16
 *
 * @param value
 * @return bf16
 */

bf16 int_to_bf16(uint32_t value, uint8_t anchor_exp)
{
	const int int_width = 32;
	const int bf16_bias = 127;
	const int bf16_frac_bits = 7;

	bf16 result;
	result.half = 0;

	int32_t signed_val = (int32_t)value;

	if (signed_val == 0)
	{
		return result;
	}

	result.sign = (signed_val < 0);
	uint32_t mag = result.sign ? (uint32_t)(-signed_val) : (uint32_t)signed_val;

	int msb_index = 31 - __builtin_clz(mag);

	int32_t unbiased_exp = msb_index + (int32_t)anchor_exp - (int_width - 1);

	int32_t biased_exp = unbiased_exp + bf16_bias;

	if (biased_exp <= 0)
	{
		result.half = 0;
		return result;
	}

	if (biased_exp >= 255)
	{
		result.exponent = 0xFF;
		result.mantissa = 0;
		return result;
	}

	result.exponent = (uint16_t)biased_exp;

	uint32_t normalized = mag << (31 - msb_index);

	result.mantissa = (normalized >> (32 - 1 - bf16_frac_bits)) & 0x7F;
	return result;
}