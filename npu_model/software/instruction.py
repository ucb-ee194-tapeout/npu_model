from typing import TYPE_CHECKING, Callable, ClassVar, Protocol, runtime_checkable, Any, TypeIs

from ..isa_types import ScalarReg, ScalarRegL, ExponentReg, ExponentRegL, MatrixReg, MatrixRegL, Accumulator, AccumulatorL, WeightBuffer, WeightBufferL, Opcode, EXU
from ..isa_types import Named, AsmError

if TYPE_CHECKING:
    from ..hardware.arch_state import ArchState


@runtime_checkable
class Instruction(Protocol):
    """
    Structural type for concrete instruction classes that inherit from both
    Instruction and InstructionPattern. Use this instead of casting between the two.
    """
    mnemonic: ClassVar[str]
    params:   ClassVar[list[Named]]
    opcode:   ClassVar[Opcode]
    exu:      ClassVar[EXU]

    def to_bytecode(self) -> int: ...
    def exec(self, state: ArchState) -> None: ...
    def serialize(self) -> str: ...

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = ...) -> Instruction: ...

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]: ...

    @classmethod
    def num_args(cls) -> int: ...

    @classmethod
    def num_toks(cls) -> int: ...


def is_instruction(cls: type[Any]) -> TypeIs[type[Instruction]]:
    return hasattr(cls, 'mnemonic')        \
           and hasattr(cls, 'params')      \
           and hasattr(cls, 'opcode')      \
           and hasattr(cls, 'exu')         \
           and hasattr(cls, 'to_bytecode') \
           and hasattr(cls, 'exec')        \
           and hasattr(cls, 'serialize')   \
           and hasattr(cls, 'from_asm')    \
           and hasattr(cls, 'lint')        \
           and hasattr(cls, 'num_args')    \
           and hasattr(cls, 'num_toks')

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