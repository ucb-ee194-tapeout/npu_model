from ..isa import Instruction
from ..isa_types import ScalarReg, ScalarRegL, ExponentReg, ExponentRegL, MatrixReg, MatrixRegL, Accumulator, AccumulatorL, WeightBuffer, WeightBufferL


# Utility Functions to be used for writing programs:
def x(val: ScalarRegL) -> ScalarReg:
    """
    Casts an integer as a Scalar Register

    Args:
        val: an integer from 0–31 to cast
    
    Returns:
        A reference to the corresponding scalar register
    
    Raises:
        ValueError: if val is not 0–31.
    """
    return ScalarReg(val)

def e(val: ExponentRegL) -> ExponentReg:
    """
    Casts an integer as a Exponent Register

    Args:
        val: an integer from 0–31 to cast
    
    Returns:
        A reference to the corresponding exponent register
    
    Raises:
        ValueError: if val is not 0–31.
    """
    return ExponentReg(val)

def m(val: MatrixRegL) -> MatrixReg:
    """
    Casts an integer as a Matrix Register

    Args:
        val: an integer from 0–63 to cast
    
    Returns:
        A reference to the corresponding exponent register
    
    Raises:
        ValueError: if val is not 0–63.
    """
    return MatrixReg(val)

def w(val: WeightBufferL) -> WeightBuffer:
    """
    Casts an integer as a Matrix Register

    Args:
        val: an integer from 0–63 to cast
    
    Returns:
        A reference to the corresponding exponent register
    
    Raises:
        ValueError: if val is not 0–63.
    """
    return WeightBuffer(val)

def acc(val: AccumulatorL) -> Accumulator:
    """
    Casts an integer as a Matrix Register

    Args:
        val: an integer from 0–63 to cast
    
    Returns:
        A reference to the corresponding exponent register
    
    Raises:
        ValueError: if val is not 0–63.
    """
    return Accumulator(val)

class Uop():
    """
    A dynamic instruction instance that is executing in the simulation
    """

    _next_id: int = 0

    def __init__(self, insn: Instruction) -> None:
        self.id = Uop._next_id
        Uop._next_id += 1
        self.insn = insn

        self.dispatch_delay: int = 0
        """the number of dispatch stalling cycles left"""
        self.execute_delay: int = 0
        """the number of execute stalling cycles left"""