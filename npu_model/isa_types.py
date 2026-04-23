"""
Convenience Types that ensure that programs have valid inputs.

You should not have to import anything from this module to write programs. If you are attempting
to write programs, see the utility functions in isa.py

In this namespace:
- Types that end in "L" are meant to be used for static type checking (i.e. User Input)
- Types that end in "T" are type types (i.e. a union of types). They're useful for allowing multiple
  types to be passed into a function (the class itself, not objects instantiated from the class).
- Types that do not end in "L" or "T" are meant to be used to dynamically check for correctness at runtime.
  Each type implements:
  - a constructor that converts a string or int into the type
  - `is_signed()`: returns True if the literal falls into the signed range
  - `is_unsigned()`: returns True if the literal falls into the unsigned range
  - `accepts(val: str | int)`: returns True if the calling the constructor would succeed on the input.
"""

from enum import StrEnum
from typing import Literal, TypeVar, Callable
import re

class AsmError(ValueError):
    def __init__(self, message: str, *, token_index: int = 0):
        super().__init__(message)
        self.token_index = token_index  # which operand position failed (0 = mnemonic)


class EXU(StrEnum):
    CORE = "Core"
    IFU = "InstructionFetch"
    IDU = "InstructionDecode"
    SCALAR = "ScalarExecutionUnit"
    VECTOR = "VectorExecutionUnit"
    MATRIX_SYSTOLIC = "MatrixExecutionUnitSystolic"
    MATRIX_INNER = "MatrixExecutionUnitInner"
    DMA = "DmaExecutionUnit"
    LSU = "LoadStoreUnit"


class BoundedInt(int):
    """
    A helper class used to define the special types in this file.
    Ironically, by default this int is unbounded.

    A bound can be imposed by setting `lower_bound` and `upper_bound` in subclasses.
    Optionally, a `unsigned_lower_bound` and `signed_upper_bound` can be set, for use with
    the `is_signed`, and `is_unsigned` functions.

    NOTE:
    - Lower bound is inclusive, upper bound is *exclusive* (python-style, not Verilog-style)
    - `lower_bound` and `upper_bound` should represent the true bounds of the value (signed/unsigned
      should be a subset)
    - For points between unsigned_lower_bound and signed_upper_bound, both `is_signed` and
      `is_unsigned` will return True. This is intended behavior, since the value is safe in
      either case.
    """

    lower_bound = 0
    upper_bound = 0

    unsigned_lower_bound = 0
    signed_upper_bound = 0

    fmt: str = ""

    def is_signed(self) -> bool:
        return self.lower_bound <= self < self.signed_upper_bound

    def isreturn_unsigned(self) -> bool:
        return self.unsigned_lower_bound <= self < self.upper_bound

    @classmethod
    def autocomplete(cls) -> list[str]:
        return []

    @classmethod
    def format_arg(cls, role: str):
        """
        Returns a human-readable form for display in an instruction pattern.
        """
        return f"<{cls.fmt}>"

    @classmethod
    def lint(cls, val: str | int, role: str = "", tok_idx: int = 0) -> list[AsmError]:
        """
        Checks if a given string or int can be cast correctly.

        Args:
            val: The integer value to validate.

        Returns:
            True if val is in-range, False otherwise.
        """
        if isinstance(val, str):
            try:
                val = int(val, 0)
            except ValueError:
                return [
                    AsmError(
                        f"Expected integer for {role if role != '' else cls.fmt}, got '{val}'",
                        token_index=tok_idx,
                    )
                ]

        if cls.lower_bound <= val < cls.upper_bound:
            return []

        return [
            AsmError(
                f"{cls.fmt} value {str(val)} is out of range [{cls.lower_bound}, {cls.upper_bound})",
                token_index=tok_idx,
            )
        ]

    def __new__(cls, val: int | str):
        if len(err := cls.lint(val)) != 0:
            raise ExceptionGroup(f"{cls.__name__} failed to initialize:", err)

        if isinstance(val, str):
            val = int(val, 0)

        return super().__new__(cls, val)


# Opcode Registers
type OpcodeL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]
class Opcode(BoundedInt):
    """
    Represents a 7-bit opcode value.

    This class validates and coerces integer or string values into valid opcode
    values within the 7-bit range (0-127).
    """

    lower_bound = 0
    upper_bound = 128


# Funct2 Registers
type Funct2L = Literal[0, 1, 2, 3]


class Funct2(BoundedInt):
    """
    Represents a 2-bit funct2 value.

    This class validates and coerces integer or string values into valid funct2
    values within the 2-bit range (0-3).
    """

    lower_bound = 0
    upper_bound = 4


# Funct3 Registers
type Funct3L = Literal[0, 1, 2, 3, 4, 5, 6, 7]


class Funct3(BoundedInt):
    """
    Represents a 3-bit Funct3 value.

    This class validates and coerces integer or string values into valid Funct3
    values within the 3-bit range (0-7).
    """

    lower_bound = 0
    upper_bound = 8


# Funct7 Registers
type Funct7L = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]
class Funct7(BoundedInt):
    """
    Represents a 7-bit Funct7 value.

    This class validates and coerces integer or string values into valid Funct7
    values within the 7-bit range (0-127).
    """

    lower_bound = 0
    upper_bound = 128


class RegBase(BoundedInt):
    reg_name: str = "base"

    @classmethod
    def autocomplete(cls) -> list[str]:
        return [f"{cls.fmt}{i}" for i in range(cls.lower_bound, cls.upper_bound)]

    @classmethod
    def format_arg(cls, role: str):
        return f"{cls.fmt}<{role}>"

    @classmethod
    def lint(cls, val: str | int, role: str = "", tok_idx: int = 0) -> list[AsmError]:
        if isinstance(val, str):
            val = val.lower()
            if not val.startswith(cls.fmt):
                return [
                    AsmError(
                        f"Expected {cls.reg_name} ({cls.fmt}{cls.lower_bound}–{cls.upper_bound - 1}) for {role}, got '{val}'",
                        token_index=tok_idx,
                    )
                ]
            val = val[len(cls.fmt):]

        try:
            val = int(val)
        except:
            return [
                AsmError(
                    f"Malformed {cls.reg_name}: expected {cls.fmt}<{role}>, got '{val}'",
                    token_index=tok_idx,
                )
            ]

        if cls.lower_bound <= val < cls.upper_bound:
            return []

        return [
            AsmError(
                f"{cls.reg_name.capitalize()} '{val}' out of range ({cls.fmt}{cls.lower_bound}–{cls.upper_bound - 1})",
                token_index=tok_idx,
            )
        ]

    def __new__(cls, val: str | int):
        if isinstance(val, str):
            val = val.lower()
            if not val.startswith(cls.fmt):
                raise ValueError(
                    f"Attempted to intialize a malformed {cls.reg_name} (missing '{cls.fmt}'): {val}"
                )
            val = int(val[len(cls.fmt):])

        return super().__new__(cls, val)


# Scalar Registers
type ScalarRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
class ScalarReg(RegBase):
    """
    Represents a 5-bit reference to a Scalar Register.

    This class validates and coerces integer or string values into valid ScalarReg
    values within the 5-bit range (0-31).

    Strings passed in must begin with `x` and be in base-10 (i.e. `x31`)
    """

    lower_bound = 0
    upper_bound = 32
    fmt = "x"
    reg_name = "scalar register"

    @classmethod
    def lint(cls, val: str | int, role: str = "", tok_idx: int = 0) -> list[AsmError]:
        if isinstance(val, str):
            if re.match(r"\b(zero|ra|sp|gp|tp|t[0-6]|s0|fp|s[1-9]|s1[0-1]|a[0-7])\b", val) != None:
                return []
        return super().lint(val, role=role, tok_idx=tok_idx)

    def __new__(cls, val: str | int):
        # We want scalar registers to support using ABI names in asm
        if isinstance(val, str):
            val = val.lower()

        match val:
            case "zero": return super().__new__(cls, 0)
            case "ra": return super().__new__(cls, 1)
            case "sp": return super().__new__(cls, 2)
            case "gp": return super().__new__(cls, 3)
            case "tp": return super().__new__(cls, 4)
            case "t0" | "t1" | "t2": return super().__new__(cls, 5 + int(val[1:]))
            case "s0" | "fp": return super().__new__(cls, 8)
            case "s1": return super().__new__(cls, 9)
            case "a0" | "a1" | "a2" | "a3" | "a4" | "a5" | "a6" | "a7":
                return super().__new__(cls, 10 + int(val[1:]))
            case "s2" | "s3" | "s4" | "s5" | "s6" | "s7" | "s8" | "s9" | "s10" | "s11":
                return super().__new__(cls, 16 + int(val[1:]))
            case "t3" | "t4" | "t5" | "t6":
                return super().__new__(cls, 25 + int(val[1:]))                
            case other: return super().__new__(cls, other)
        


# Exponent Registers
type ExponentRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
class ExponentReg(RegBase):
    """
    Represents a 5-bit reference to a Exponent Register.

    This class validates and coerces integer or string values into valid ExponentReg
    values within the 5-bit range (0-31).

    Strings passed in must begin with `e` and be in base-10 (i.e. `e31`)
    """

    lower_bound = 0
    upper_bound = 32
    fmt = "e"
    reg_name = "exponent register"


# Matrix Registers
type MatrixRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
class MatrixReg(RegBase):
    """
    Represents a 6-bit reference to a Matrix Register.

    This class validates and coerces integer or string values into valid MatrixReg
    values within the 6-bit range (0-63).

    Strings passed in must begin with `m` and be in base-10 (i.e. `m63`)
    """

    lower_bound = 0
    upper_bound = 64
    fmt = "m"
    reg_name = "matrix register"


type AccumulatorL = Literal[0, 1]


class Accumulator(RegBase):
    """
    Represents a 1-bit reference to an accumulator buffer.

    This class validates and coerces integer or string values into valid MatrixReg
    values within the 1-bit range (0-1).

    Strings passed in must begin with `a` and be in base-10 (i.e. `a0`)
    """

    lower_bound = 0
    upper_bound = 2
    fmt = "acc"
    reg_name = "accumulator"


type WeightBufferL = Literal[0, 1]
class WeightBuffer(RegBase):
    """
    Represents a 1-bit reference to an accumulator buffer.

    This class validates and coerces integer or string values into valid MatrixReg
    values within the 1-bit range (0-1).

    Strings passed in must begin with `w` and be in base-10 (i.e. `w0`)
    """

    lower_bound = 0
    upper_bound = 2
    fmt = "w"
    reg_name = "weight buffer"


# Register
RegisterT = TypeVar("RegisterT", type[ScalarReg], type[ExponentReg], type[MatrixReg])


# Shamt
class Shamt(BoundedInt):
    """
    Represents a 5-bit Immediate.

    This class validates and coerces integer or string values into valid Shamt
    values within the 5-bit range (0-31).
    """

    lower_bound = 0
    upper_bound = 32
    fmt = "shamt"


# Imm12
class Imm12(BoundedInt):
    """
    Represents a 12-bit Immediate.

    This class validates and coerces integer or string values into valid Imm12
    values within the 12-bit range (0-4095).
    """

    lower_bound = -2048
    unsigned_lower_bound = 0
    signed_upper_bound = 2048  # remember, top of range is exclusive
    upper_bound = 4096
    fmt = "imm12"


# SBImm12
class SBImm12(BoundedInt):
    """
    Represents a 12-bit signed byte offset for SBType branch and JALR instructions.

    Targets must be 2-byte aligned (even), matching RISC-V SB encoding.
    The assembler produces byte offsets (word_offset * 4) for label resolution.
    """

    lower_bound = -2048
    unsigned_lower_bound = 0
    signed_upper_bound = 2048  # remember, top of range is exclusive
    upper_bound = 4096
    fmt = "sbimm12"

    @classmethod
    def lint(cls, val: str | int, role: str = "", tok_idx: int = 0) -> list[AsmError]:
        exp = super().lint(val, role, tok_idx)
        if len(exp) != 0:
            return exp

        val = int(val, 0) if isinstance(val, str) else val
        if val % 2 == 0:
            return []

        return [
            AsmError(f"Branch offset must be even (got {val})", token_index=tok_idx)
        ]


# Imm16
class Imm16(BoundedInt):
    """
    Represents a 16-bit Immediate.

    This class validates and coerces integer or string values into valid Imm16
    values within the 16-bit range (0-65535).
    """

    lower_bound = -32768
    unsigned_lower_bound = 0
    signed_upper_bound = 32768  # remember, top of range is exclusive
    upper_bound = 65536
    fmt = "imm16"


# Imm20
class Imm20(BoundedInt):
    """
    Represents a 20-bit Immediate.

    This class validates and coerces integer or string values into valid Imm20
    values within the 20-bit range (0-1048575).
    """

    lower_bound = -524288
    unsigned_lower_bound = 0
    signed_upper_bound = 524288  # remember, top of range is exclusive
    upper_bound = 1048576
    fmt = "imm20"

# Imm32 — not real. Only used in Pseudoinstrs
class Imm32(BoundedInt):
    lower_bound = -2147483648
    unsigned_lower_bound = 0
    signed_upper_bound = 2147483648  # remember, top of range is exclusive
    upper_bound = 4294967296
    fmt = "imm32"

class Named():
    """
    Represents a typed value with a specific name, for use in human-readable instruction patterns.
    """
    is_label = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    def __init__(self, inner: type[BoundedInt], name: str, repr: str = "", label_support: bool = False):
        self.inner = inner
        self.name = name
        self.repr = repr if repr != "" else name
        self.label_support = label_support

    def autocomplete(self, labels: list[str]):
        return labels if self.label_support else self.inner.autocomplete()

    def lint(self, val: str | int, labels: list[str], tok_idx: int, allow_label: bool) -> list[AsmError]:
        if self.label_support and allow_label and isinstance(val, str) and self.is_label.match(val) != None:
            return [] if val in labels else [AsmError(f"Undefined label '{val}'", token_index=2)]
        return self.inner.lint(val, role=self.name, tok_idx=tok_idx)
    
    def format_arg(self, allow_label: bool = False):
        return f"{self.inner.format_arg(self.name)}{' or label' if self.label_support and allow_label else ''}"
    
    def parse_token(self, val: str | int, resolve: Callable[[str], int]) -> dict[str, BoundedInt]:
        """Parse a single token according to this Named parameter's inner type."""
        if self.label_support and isinstance(val, str) and self.is_label.match(val):
            return {self.repr: self.inner(resolve(val))}
        return {self.repr: self.inner(val)}


class Bundled(Named):
    """
    Represents a bundled immediate and register.
    """
    scalar_offset_fmt = re.compile(
        r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(x(\d+)\)"
    )
    exponent_offset_fmt = re.compile(
        r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(e(\d+)\)"
    )
    matrix_offset_fmt = re.compile(
        r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(m(\d+)\)"
    )


    def __init__(self, reg: Named, imm: Named):
        self.reg = reg
        self.imm = imm

    def lint(self, val: str | int, labels: list[str], tok_idx: int, allow_label: bool) -> list[AsmError]:
        if isinstance(val, int):
            raise ValueError("Provided an int for linting to a Bundle.")
        if self.reg.inner is ScalarReg:
            match = self.scalar_offset_fmt.match(val)
        elif self.reg.inner is ExponentReg:
            match = self.exponent_offset_fmt.match(val)
        elif self.reg.inner is MatrixReg:
            match = self.matrix_offset_fmt.match(val)
        else:
            raise ValueError(f"Attempted to provide a non-register format to Bundle: {self.reg.inner.__name__}")
        
        if not match:
            return [AsmError(f"Expected base+offset operand in the form {self.format_arg()}, got '{val}'", token_index=tok_idx)]
        
        imm = int(match.group(1), 0)
        reg = int(match.group(2))

        err = self.imm.lint(imm, labels, tok_idx, allow_label)
        err.extend(self.reg.lint(reg, labels, tok_idx, allow_label))
        return err
    
    def format_arg(self, allow_label: bool = False):
        return f"{self.imm.format_arg(allow_label)}({self.reg.format_arg(allow_label)})"

    def parse_token(self, val: str | int, resolve: Callable[[str], int]) -> dict[str, BoundedInt]:
        """Parse a single token according to this Named parameter's inner type."""
        if isinstance(val, int):
            raise ValueError("Provided an int for linting to a Bundle.")
        if self.reg.inner is ScalarReg:
            match = self.scalar_offset_fmt.match(val)
        elif self.reg.inner is ExponentReg:
            match = self.exponent_offset_fmt.match(val)
        elif self.reg.inner is MatrixReg:
            match = self.matrix_offset_fmt.match(val)
        else:
            raise ValueError(f"Attempted to provide a non-register format to Bundle: {self.reg.inner.__name__}")
        
        if not match:
            raise ValueError(f"Expected base+offset operand in the form {self.format_arg()}, got '{val}'")
        
        imm = int(match.group(1), 0)
        reg = int(match.group(2))
        
        return self.imm.parse_token(imm, resolve) | self.reg.parse_token(reg, resolve)

ImmediateT = TypeVar("ImmediateT", type[Imm12], type[SBImm12], type[Imm16], type[Imm20])
