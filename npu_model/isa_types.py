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
from typing import TypeVar, Literal
from enum import StrEnum

class EXU(StrEnum):
    CORE            = "Core"
    IFU             = "InstructionFetch"
    IDU             = "InstructionDecode"
    SCALAR          = "ScalarExecutionUnit"
    VECTOR          = "VectorExecutionUnit"
    MATRIX_SYSTOLIC = "MatrixExecutionUnitSystolic"
    MATRIX_INNER    = "MatrixExecutionUnitInner"
    DMA             = "DmaExecutionUnit"

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
    lower_bound = float('-inf')
    upper_bound = float('inf')

    unsigned_lower_bound = 0
    signed_upper_bound   = 0

    def is_signed(self) -> bool:
        return self.lower_bound <= self < self.signed_upper_bound

    def is_unsigned(self) -> bool:
        return self.unsigned_lower_bound <= self < self.upper_bound

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        """
        Checks if a given string or int can be cast correctly.

        Args:
            val: The integer value to validate.

        Returns:
            True if val is in-range, False otherwise.
        """
        if isinstance(val, str):
            val = int(val, 0)
        
        return cls.lower_bound <= val < cls.upper_bound
    
    def __new__(cls, val: int | str):
        if not cls.accepts(val):
            raise ValueError(f"Invalid value provided for {cls.__name__}: {val}")

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
type Funct2L = Literal[0,1,2,3]
class Funct2(BoundedInt):
    """
    Represents a 2-bit funct2 value.
    
    This class validates and coerces integer or string values into valid funct2
    values within the 2-bit range (0-3).
    """
    lower_bound = 0
    upper_bound = 4

# Funct3 Registers
type Funct3L = Literal[0,1,2,3,4,5,6,7]
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


# Scalar Registers
type ScalarRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
class ScalarReg(BoundedInt):
    """
    Represents a 5-bit reference to a Scalar Register.
    
    This class validates and coerces integer or string values into valid ScalarReg
    values within the 5-bit range (0-31).

    Strings passed in must begin with `x` and be in base-10 (i.e. `x31`)
    """
    lower_bound = 0
    upper_bound = 32

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        if isinstance(val, str):
            if not val.startswith("x"):
                raise ValueError(f"Attempted to intialize a malformed scalar register (missing 'x'): {val}")
            
            val = int(val[1:])
        
        return cls.lower_bound <= val <= cls.upper_bound

    def __new__(cls, val: str | int):
        if isinstance(val, str):
            if not val.startswith("x"):
                raise ValueError(f"Attempted to intialize a malformed scalar register (missing 'x'): {val}")
            val = int(val[1:])

        return super().__new__(cls, val)

# Exponent Registers
type ExponentRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
class ExponentReg(BoundedInt):
    """
    Represents a 5-bit reference to a Exponent Register.
    
    This class validates and coerces integer or string values into valid ExponentReg
    values within the 5-bit range (0-31).

    Strings passed in must begin with `e` and be in base-10 (i.e. `e31`)
    """
    lower_bound = 0
    upper_bound = 32

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        if isinstance(val, str):
            if not val.startswith("e"):
                raise ValueError(f"Attempted to intialize a malformed exponent register (missing 'e'): {val}")
            
            val = int(val[1:])
        
        return cls.lower_bound <= val <= cls.upper_bound
    
    def __new__(cls, val: str | int):
        if isinstance(val, str):
            if not val.startswith("e"):
                raise ValueError(f"Attempted to intialize a malformed exponent register (missing 'e'): {val}")
            val = int(val[1:])

        return super().__new__(cls, val)

# Matrix Registers
type MatrixRegL = Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
class MatrixReg(BoundedInt):
    """
    Represents a 6-bit reference to a Matrix Register.
    
    This class validates and coerces integer or string values into valid MatrixReg
    values within the 6-bit range (0-63).

    Strings passed in must begin with `m` and be in base-10 (i.e. `m63`)
    """
    lower_bound = 0
    upper_bound = 64

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        if isinstance(val, str):
            if not val.startswith("m"):
                raise ValueError(f"Attempted to intialize a malformed matrix register (missing 'm'): {val}")
            
            val = int(val[1:])
        
        return cls.lower_bound <= val <= cls.upper_bound
        
    def __new__(cls, val: str | int):
        if isinstance(val, str):
            if not val.startswith("m"):
                raise ValueError(f"Attempted to intialize a malformed matrix register (missing 'm'): {val}")
            val = int(val[1:])

        return super().__new__(cls, val)

type AccumulatorL = Literal[0,1]
class Accumulator(BoundedInt):
    """
    Represents a 1-bit reference to an accumulator buffer.
    
    This class validates and coerces integer or string values into valid MatrixReg
    values within the 1-bit range (0-1).

    Strings passed in must begin with `a` and be in base-10 (i.e. `a0`)
    """
    lower_bound = 0
    upper_bound = 2

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        if isinstance(val, str):
            if not val.startswith("a"):
                raise ValueError(f"Attempted to intialize a malformed accumulator (missing 'a'): {val}")
            
            val = int(val[1:])
        
        return cls.lower_bound <= val <= cls.upper_bound
    
    def __new__(cls, val: str | int):
        if isinstance(val, str):
            if not val.startswith("a"):
                raise ValueError(f"Attempted to intialize a malformed accumulator (missing 'a'): {val}")
            val = int(val[1:])

        return super().__new__(cls, val)
    
type WeightBufferL = Literal[0,1]
class WeightBuffer(BoundedInt):
    """
    Represents a 1-bit reference to an accumulator buffer.
    
    This class validates and coerces integer or string values into valid MatrixReg
    values within the 1-bit range (0-1).

    Strings passed in must begin with `w` and be in base-10 (i.e. `w0`)
    """
    lower_bound = 0
    upper_bound = 2

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        if isinstance(val, str):
            if not val.startswith("w"):
                raise ValueError(f"Attempted to intialize a malformed weight buffer (missing 'w'): {val}")
            
            val = int(val[1:])
        
        return cls.lower_bound <= val <= cls.upper_bound

    def __new__(cls, val: str | int):
        if isinstance(val, str):
            if not val.startswith("w"):
                raise ValueError(f"Attempted to intialize a malformed weight buffer (missing 'w'): {val}")
            val = int(val[1:])

        return super().__new__(cls, val)
# Register
RegisterT = TypeVar('RegisterT', type[ScalarReg], type[ExponentReg], type[MatrixReg])

# Shamt
class Shamt(BoundedInt):
    """
    Represents a 5-bit Immediate.
    
    This class validates and coerces integer or string values into valid Shamt
    values within the 5-bit range (0-31).
    """
    lower_bound = 0
    upper_bound = 32

# Imm12
class Imm12(BoundedInt):
    """
    Represents a 12-bit Immediate.
    
    This class validates and coerces integer or string values into valid Imm12
    values within the 12-bit range (0-4095).
    """
    lower_bound = -2048
    unsigned_lower_bound = 0
    signed_upper_bound = 2048 # remember, top of range is exclusive
    upper_bound = 4096
    
# SBImm12
class SBImm12(BoundedInt):
    """
    Represents a 12-bit Immediate.
    
    This class validates and coerces integer or string values into valid SBImm12
    values within the 12-bit range (0-4095).
    """
    lower_bound = -2048
    unsigned_lower_bound = 0
    signed_upper_bound = 2048 # remember, top of range is exclusive
    upper_bound = 4096

    @classmethod
    def accepts(cls, val: str | int) -> bool:
        val = int(val, 0) if isinstance(val, str) else val
        if val % 2 == 1:
            raise ValueError("Attempted to initialize an SB immediate that ends in 1.")
    
        return super().accepts(val)

# Imm16
class Imm16(BoundedInt):
    """
    Represents a 16-bit Immediate.
    
    This class validates and coerces integer or string values into valid Imm16
    values within the 16-bit range (0-65535).
    """
    lower_bound = -32768
    unsigned_lower_bound = 0
    signed_upper_bound = 32768 # remember, top of range is exclusive
    upper_bound = 65536
    
# Imm20
class Imm20(BoundedInt):
    """
    Represents a 20-bit Immediate.
    
    This class validates and coerces integer or string values into valid Imm20
    values within the 20-bit range (0-1048575).
    """
    lower_bound = -524288
    unsigned_lower_bound = 0
    signed_upper_bound = 524288 # remember, top of range is exclusive
    upper_bound = 1048576

ImmediateT = TypeVar("ImmediateT", type[Imm12], type[SBImm12], type[Imm16], type[Imm20])