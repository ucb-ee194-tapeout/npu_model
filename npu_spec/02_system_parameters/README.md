# System Parameters

This section covers the overall system design parameters and definition of key terms.

## Data Types

The data types that are used in this document and operated by the Accelerator include:

- Boolean (1-bit) number
- Unsigned 32-bit integer number
- Signed 32-bit integer number
- 32-bit floating point number
- 16-bit floating point number
- 16-bit Google Brain floating point number
- 8-bit floating point number

The standard name, short hand name, notations, and size of these data types are defined in the table below:

| Name | Short Hand Name | Notation | Size (bits) |
|------|-----------------|----------|-------------|
| Boolean | boolean | bool | 1 |
| Unsigned 32-bit integer | uint32 | u32 | 32 |
| Signed 32-bit integer | int32 | i32 | 32 |
| IEEE 754 Single Precision Floating Point (binary32) | float32 | fp32 | 32 |
| IEEE 754 Half Precision Floating Point (binary16) | float16 | fp16 | 16 |
| Brain Floating Point | bfloat16 | bf16 | 16 |
| OCP 8-bit Floating Point (E4M3) | float8_e4m3 | fp8 | 8 |
| OCP 8-bit Floating Point (E5M2) | float8_e5m2 | fp8_e5m2 | 8 |


### Float32 Type

![](/docs/images/datatype_fp32.png)

The IEEE 754 float32 format (officially known as binary32) allocates its 32 bits as follows: 

- 1 sign bit (S): Determines whether the number is positive (0) or negative (1).
- 8 exponent bits (E): Store the exponent value using an offset-binary representation with a bias of 127.
- 23 mantissa (or significand/fraction) bits (M): Represent the fractional part of the number, with an implicit leading bit (usually 1 for normal numbers).

### Float16 Type

![](/docs/images/datatype_fp16.png)

The IEEE 754 float16 format (officially known as binary16) allocates its 16 bits as follows:
- 1 sign bit (S): Determines whether the number is positive (0) or negative (1).
- 5 exponent bits (E): Store the exponent value using an offset-binary representation with a bias of 15.
- 10 mantissa (or significand/fraction) bits (M): Represent the fractional part of the number, with an implicit leading bit (usually 1 for normal numbers).

### BFloat16 Type

![](/docs/images/datatype_bf16.png)

The bfloat16 format (Brain Floating Point) allocates its 16 bits as follows:
- 1 sign bit (S): Determines whether the number is positive (0) or negative (1).
- 8 exponent bits (E): Store the exponent value using an offset-binary representation with a bias of 127 (the same exponent size and bias as float32).
- 7 mantissa (or significand/fraction) bits (M): Represent the fractional part of the number, with an implicit leading bit (usually 1 for normal numbers).

### Float8 E4M3 Type

![](/docs/images/datatype_fp8_e4m3.png)

The FP8 e4m3 format allocates its 8 bits as follows:
- 1 sign bit (S): Determines whether the number is positive (0) or negative (1).
- 4 exponent bits (E): Store the exponent value using an offset-binary representation with a bias of 7.
- 3 mantissa (or significand/fraction) bits (M): Represent the fractional part of the number, with an implicit leading bit (usually 1 for normal numbers).


### Float8 E5M2 Type

![](/docs/images/datatype_fp8_e5m2.png)

The FP8 e5m2 format allocates its 8 bits as follows:
- 1 sign bit (S): Determines whether the number is positive (0) or negative (1).
- 5 exponent bits (E): Store the exponent value using an offset-binary representation with a bias of 15.
- 2 mantissa (or significand/fraction) bits (M): Represent the fractional part of the number, with an implicit leading bit (usually 1 for normal numbers).

## Design Parameters

Table below summarizes the key design parameters.

| Parameter | Value | Unit | Description |
|---|---|---|---|
| **SoC Parameters** |  |  |  |
| SERTL_WIDTH | 32 | bits | Number of off-chip serial TileLink pins in one direction |
| SERTL_FREQ | 200 | MHz | Off-chip serial TileLink frequency |
| CORE_FREQ | 400 | MHz | Core clock frequency |
| **Accelerator Datapath Parameters** |  |  |  |
| ML | 2 | - | Number of matrix execution units |
| NL | 16 | - | Number of inner-product trees within each MXU |
| DL | 32 | - | Number of elements reduced by each inner-product tree |
| DVL | 16 | - | Number of elements processed by VPU in parallel |
| MT | 64 | - | Depth (number of rows) of the matrix register file |
| NUM_MREG | 64 | - | Number of matrix registers |
| NUM_WB | 2 | - | Number of weight buffer entries |

The total off-chip memory bandwidth is given by `SERTL_WIDTH * SERTL_FREQ / 8 (MBps)`.
The total computation throughput of MAC operation is given by `2 * ML * NL * DL (FLOPs/cycle)`.

> **Design Rationale**
>
> In our design, MT corresponds to the latency of arithmetic instructions: each instruction at least takes MT cycles to read from the registers, process, and write back. To reduce the required instruction throughput on the front end, this number should be as large as possible. However, this number also corresponds to the sequence length tilling size of the workload (on the M dimension). During action model inference, the sequence length is 51. Therefore, setting the value larger than 64 would mean a large amount of cycles wasted.
