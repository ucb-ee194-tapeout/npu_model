import inspect

from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import Callable, ClassVar, Self, Any, get_type_hints

from .isa import UJType, SBType
from .isa_types import Named, Bundled, AsmError, BoundedInt, ScalarReg, ExponentReg, MatrixReg, WeightBuffer, Accumulator, Shamt, Imm12, SBImm12, Imm16, Imm20, RegBase

# - Named Args ----------------------------------------------
# Used to significantly clean up linting and assembling code
# -----------------------------------------------------------
# Scalar
x_rs2    = Named(ScalarReg, "rs2")
x_rs1    = Named(ScalarReg, "rs1")
x_rd     = Named(ScalarReg, "rd")

# Exponent
e_rd     = Named(ExponentReg, "rd")
e_es1    = Named(ExponentReg, "es1")

# Matrix
m_vd     = Named(MatrixReg, "vd")
m_vs1    = Named(MatrixReg, "vs1")
m_vs2    = Named(MatrixReg, "vs2")

# Weight Buffer
w_vd     = Named(WeightBuffer, "vd")
w_vs2    = Named(WeightBuffer, "vs2")

# Accumulator
acc_vd   = Named(Accumulator, "vd")
acc_vs2  = Named(Accumulator, "vs2")

# Immediates and Labels
shamt    = Named(Shamt, "shift amount", "imm")
imm12    = Named(Imm12, "12-bit immediate", "imm")
sbimm12  = Named(SBImm12, "12-bit SB immediate or label", "imm", True)
imm16    = Named(Imm16, "16-bit immediate", "imm")
imm20    = Named(Imm20, "20-bit immediate", "imm")
imm20_lb = Named(Imm20, "20-bit immediate or label", "imm", True)

# Offset-Immediate formats
x_rs1_imm12 = Bundled(x_rs1, imm12)

# Util functions
def _wrong_cnt_error(
    mnemonic: str, expected: int, actual: int, operand_fmt: str
) -> AsmError:
    return AsmError(
        f"'{mnemonic}' expects {expected - 1} operand{'s' if expected != 2 else ''} ({operand_fmt}), got {actual - 1}",
        token_index=0,
    )

class InstructionPatternMeta(ABCMeta):
    """
    Metaclass that checks that from_asm will produce a valid output.
    Errors if params don't expand out into the args necessary for __init__.
    """
    params: ClassVar[list[Named]]
    
    def __new__(mcs,name: str,bases: tuple[type, ...],namespace: dict[str, Any],**kwargs: Any) -> type:
        params: list[Named] = namespace.get('params', [])
        
        # Extract parameter names and types from params
        param_specs: list[tuple[str, type]] = []
        for param in params:
            if isinstance(param, Bundled):
                param_specs.append((param.reg.repr, param.reg.inner))
                param_specs.append((param.imm.repr, param.imm.inner))
            else:
                param_specs.append((param.repr, param.inner))
        
        param_names: list[str] = [name for name, _ in param_specs]
        
        # Check if __init__ is manually defined
        init_method: Any = namespace.get('__init__')
        
        if init_method is not None:
            # Validate that manual __init__ matches params
            sig: inspect.Signature = inspect.signature(init_method)
            init_params: list[str] = list(sig.parameters.keys())[1:]  # Skip 'self'
            
            if set(init_params) != set(param_names):
                raise TypeError(
                    f"Class {name}: __init__ parameters {init_params} "
                    f"don't match params names {param_names}"
                )
                           
            # Validate that all args subclass int
            hints: dict[str, type] = get_type_hints(init_method)
            for param_name in init_params:
                typ: type = hints[param_name]
                if not issubclass(typ,int):
                    raise TypeError(
                        f"Class {name}: {param_name} type {typ} doesn't subclass int."
                    )
        return super().__new__(mcs, name, bases, namespace, **kwargs)

class InstructionPattern(ABC, metaclass=InstructionPatternMeta):
    """
    Represents the ISA arguments for a specific instruction.

    This abstract class serves as the interface for parsing external assembly
    and providing the constructor for the Intermediate Representation (IR).
    It cannot be instantiated directly.

    All instruction patterns must define:
    - mnemonic (str): The name of the instruction. This is inferred by magic when
      paired with an InstructionType (which usees __init_subclass__ to set it.)
    - params (list[str]): The human-readable param format (for linting and errors)
    """

    mnemonic: ClassVar[str] = NotImplemented
    params: ClassVar[list[Named]] = []

    @classmethod
    def format_params(cls):
        return map(lambda x: x.format_arg(issubclass(cls, (UJType, SBType))), cls.params)

    @classmethod
    def num_args(cls):
        return len(cls.params)
    
    @classmethod
    def num_toks(cls):
        return len(cls.params) + 1

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        """
        Parse assembly instruction tokens and instantiate an instruction if the given set
        of tokens represent a valid instance of the instruction. Should be used on the
        instruction class itself (i.e. `li.from_asm`) NOT on the InstructionPattern class.

        Args:
            tokens: List of assembly instruction tokens (e.g., `["add", "x1", "x2", "x3"]`).
            resolve: Optionally, a label-resolution function if labels are supported.

        Returns:
            An instance of an Instruction parsed from the tokens.

        Raises:
            ValueError: When an invalid set of tokens is provided for a specific instruction.
            NotImplementedError: When called on the InstructionPattern class rather than a valid subclass.
        """
        if tokens[0] != cls.mnemonic:
            raise ValueError(f'Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.')
        
        kwargs: dict[str, BoundedInt] = {}
        for i, param in enumerate(cls.params):
            kwargs = kwargs | param.parse_token(tokens[i + 1], resolve)
        return cls(**kwargs)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        """
        Validate whether a set of tokens can be used to instantiate an Instruction. Should be used
        on the instruction class itself (i.e. `li.lint`) NOT on the InstructionPattern class.

        Args:
            tokens: List of assembly instruction tokens (e.g., `["add", "x1", "x2", "x3"]`).
            labels: A list of labels that are available in the assembly.

        Returns:
            A list of Assembly Errors. If the list is empty, there are no problems with the instruction.

        Note:
            We do not check whether the jump to the label is too large. We just assume this is fine.
            If it's too large, it should get caught at assemble time.
        """
        exceptions: list[AsmError] = []
        if len(tokens) != cls.num_toks():
            exceptions.append(
                _wrong_cnt_error(
                    cls.mnemonic,
                    cls.num_toks(),
                    len(tokens),
                    ', '.join(cls.format_params()),
                )
            )

        for i in range(1, min(len(tokens),cls.num_toks())):
            exceptions.extend(cls.params[i-1].lint(tokens[i], labels, tok_idx=i, allow_label=issubclass(cls, (UJType, SBType))))

        return exceptions
    
    def _unbundle(self, val: list[Named]) -> list[Named]:
        out: list[Named] = []
        for param in val:
            if isinstance(param, Bundled):
                out.extend([param.reg, param.imm])
            else:
                out.append(param)
        
        return out

    def _fmt_param(self, param: Named, val: BoundedInt, for_python: bool = False)-> str:        
        if issubclass(param.inner, RegBase):
            return f'{f'{param.repr}=' if for_python else ''}{param.inner.fmt}{'(' + str(val) + ')' if for_python else str(val)}'
        
        return f'{f'{param.repr}=' if for_python else ''}{str(val)}'

    def __str__(self):
        return f'{self.mnemonic} {', '.join([self._fmt_param(param, param_val, False) for param in self._unbundle(self.params) if (param_val:=self.__dict__[param.repr]) != None])}'

    def serialize(self):
        return f"{self.__class__.__name__}({', '.join([self._fmt_param(param, param_val, True) for param in self._unbundle(self.params) if (param_val:=self.__dict__[param.repr]) != None])})"

@dataclass(init=False)
class ScalarOffsetLoad(InstructionPattern):
    """
    Instruction pattern for instructions with a scalar destination register and base-offset operand.

    Matches assembly patterns of the form `instr x(rd), imm(x(rs1))`.
    This pattern is used for scalar load instructions: `lb`, `lh`, and `lw`.

    Attributes:
        rd: The destination scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    rd: ScalarReg
    imm: Imm12
    rs1: ScalarReg
    params: ClassVar[list[Named]] = [x_rd, x_rs1_imm12]

    def __init__(self, rd: ScalarReg, imm: int, rs1: ScalarReg):
        self.rd = rd
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class ExponentOffsetLoad(InstructionPattern):
    """
    Instruction pattern for instructions with a exponent destination register and base-offset operand.

    Matches assembly patterns of the form `instr e(rd), imm(x(rs1))`.
    This pattern is used for the following instructions: `seld`

    Attributes:
        rd: The destination scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    rd: ExponentReg
    imm: Imm12
    rs1: ScalarReg
    params: ClassVar[list[Named]] = [e_rd, x_rs1_imm12]

    def __init__(self, rd: ExponentReg, imm: int, rs1: ScalarReg):
        self.rd = rd
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class ScalarBaseOffsetStore(InstructionPattern):
    """
    Instruction pattern for instructions with a source and base-offset operand.

    Matches assembly patterns of the form `instr x(rs2), imm(x(rs1))`. This pattern is
    used for the following scalar instructions: `sb`, `sh`, and `sw`.

    Attributes:
        rs2: The source scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    rs2: ScalarReg
    imm: Imm12
    rs1: ScalarReg
    params: ClassVar[list[Named]] = [x_rs2, x_rs1_imm12]

    def __init__(self, rs2: ScalarReg, imm: int, rs1: ScalarReg):
        self.rs2 = rs2
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class TensorBaseOffset(InstructionPattern):
    """
    Instruction pattern for instructions with a tensor destination/source and base-offset operand.

    Matches assembly patterns of the form `instr m(vd), imm(x(rs1))`. This pattern is
    used for the following tensor instructions: `vload` and `vstore`.

    Attributes:
        vd: The destination/source tensor register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    vd: MatrixReg
    imm: Imm12
    rs1: ScalarReg
    params: ClassVar[list[Named]] = [m_vd, x_rs1_imm12]

    def __init__(self, vd: MatrixReg, imm: int, rs1: ScalarReg):
        self.vd = vd
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class ScalarImm(InstructionPattern):
    """
    Instruction pattern for instructions with a destination register and immediate operand.

    Matches assembly patterns of the form `instr x(rd), imm`. This pattern
    is used for the following instructions: `lui`, `auipc`, `jal`.

    Attributes:
        rd: The destination scalar register.
        imm: A 20-bit immediate value.
    """
    rd: ScalarReg
    imm: Imm20
    params: ClassVar[list[Named]] = [x_rd, imm20_lb]

    def __init__(self, rd: ScalarReg, imm: int):
        self.rd = rd
        self.imm = Imm20(imm)

@dataclass(init=False)
class ExponentImm(InstructionPattern):
    """
    Instruction pattern for instructions with a exponent destination register and immediate operand.

    Matches assembly patterns of the form `instr e(rd), imm`. This pattern is used for the following
    instructions: `seli`.

    Attributes:
        rd: The destination exponent register.
        imm: A 12-bit immediate value.
    """
    rd: ExponentReg
    imm: Imm12
    params: ClassVar[list[Named]] = [e_rd, imm12]

    def __init__(self, rd: ExponentReg, imm: int):
        self.rd = rd
        self.imm = Imm12(imm)

@dataclass(init=False)
class ScalarComputeImm(InstructionPattern):
    """
    Instruction pattern for instructions with a destination register, source register, and immediate operand.

    Matches assembly patterns of the form `instr x(rd), x(rs1), imm`. This pattern is
    used for scalar arithmetic and logical instructions with immediate operands.

    Attributes:
        rd: The destination scalar register.
        rs1: The source scalar register.
        imm: A 12-bit immediate value.
    """
    rd: ScalarReg
    rs1: ScalarReg
    imm: Imm12
    params: ClassVar[list[Named]] = [x_rd, x_rs1, imm12]

    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        self.imm = Imm12(imm)

@dataclass(init=False)
class JalrPattern(InstructionPattern):
    """
    Instruction pattern for JALR.

    Matches assembly patterns of the form `jalr x(rd), x(rs1), imm`.
    The immediate must be 2-byte aligned (even), matching RISC-V SB convention.

    Attributes:
        rd: The destination scalar register (return address).
        rs1: The base scalar register.
        imm: A 12-bit even immediate offset.
    """
    rd: ScalarReg
    rs1: ScalarReg
    imm: SBImm12
    params: ClassVar[list[Named]] = [x_rd, x_rs1, sbimm12]

    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        self.imm = SBImm12(imm)

@dataclass(init=False)
class ScalarComputeShamt(InstructionPattern):
    """
    Instruction pattern for the shift operator. Restricts immediate size to shamt.

    Matches assembly patterns of the form `instr x(rd), x(rs1), shamt`.

    Attributes:
        rd: The destination scalar register.
        rs1: The source scalar register.
        shamt: A 5-bit immediate value.
    """
    UPPER_IMM: int = 0b0000000
    rd: ScalarReg
    rs1: ScalarReg
    imm: Imm12
    params: ClassVar[list[Named]] = [x_rd, x_rs1, shamt]
    
    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        # Done like this on purpose so it doesn't need custom isa types.
        self.imm = Imm12((self.UPPER_IMM << 5) | Shamt(imm))

@dataclass(init=False)
class ScalarBranchImm(InstructionPattern):
    """
    Instruction pattern for branches.

    Matches assembly patterns of the form `instr x(rs1), x(rs2), imm`.

    Attributes:
        rs1: A source scalar register.
        rs2: A source scalar register.
        imm: A 12-bit immediate value.
    """

    rs1: ScalarReg
    rs2: ScalarReg
    imm: SBImm12
    params: ClassVar[list[Named]] = [x_rs1,x_rs2,sbimm12]

    def __init__(self, rs1: ScalarReg, rs2: ScalarReg, imm: int):
        self.rs1 = rs1
        self.rs2 = rs2
        self.imm = SBImm12(imm)

@dataclass(init=False)
class DirectImm(InstructionPattern):
    """
    Instruction pattern for instructions with a register and immediate operand.

    Matches assembly patterns of the form `instr m(vd), imm`. This pattern is
    used for VI instructions: `vli.all`, `vli.row`, `vli.col`, and `vli.one`.

    Attributes:
        reg: The register operand.
        imm: An immediate value.
    """
    vd: MatrixReg
    imm: Imm16
    params: ClassVar[list[Named]] = [m_vd, imm16]

    def __init__(self, vd: MatrixReg, imm: int):
        self.vd = vd
        self.imm = Imm16(imm)

@dataclass()
class ScalarComputeReg(InstructionPattern):
    """
    Instruction pattern for instructions with three scalar register operands.

    Matches assembly patterns of the form `instr x(rd), x(rs1), x(rs2)`. This pattern is
    used for scalar arithmetic and logical instructions with three register operands.

    Attributes:
        rd: The destination scalar register.
        rs1: The first source scalar register.
        rs2: The second source scalar register.
    """
    rd: ScalarReg
    rs1: ScalarReg
    rs2: ScalarReg
    params: ClassVar[list[Named]] = [x_rd, x_rs1, x_rs2]

@dataclass()
class TensorComputeUnary(InstructionPattern):
    """
    Instruction pattern for tensor instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1)`. This pattern is
    used for unary tensor operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: MatrixReg
    vs1: MatrixReg
    params: ClassVar[list[Named]] = [m_vd, m_vs1]

@dataclass()
class TensorComputeBinary(InstructionPattern):
    """
    Instruction pattern for tensor instructions with three tensor register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1), m(vs2)`. This pattern is
    used for binary tensor operations with one destination and two source tensor registers.

    Attributes:
        vd: The destination tensor register.
        vs1: The first source tensor register.
        vs2: The second source tensor register.
    """
    vd: MatrixReg
    vs1: MatrixReg
    vs2: MatrixReg
    params: ClassVar[list[Named]] = [m_vd, m_vs1, m_vs2]

@dataclass()
class TensorComputeMixed(InstructionPattern):
    """
    Instruction pattern for tensor instructions with mixed tensor and exponent register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1), e(es1)`. This pattern is
    used for tensor operations with one destination tensor register, one source tensor register,
    and one source exponent register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
        es1: The source exponent register.
    """
    vd: MatrixReg
    vs2: MatrixReg
    es1: ExponentReg
    params: ClassVar[list[Named]] = [m_vd, m_vs2, e_es1]

@dataclass()
class MXUWeightPush(InstructionPattern):
    """
    Instruction pattern for MXU data push instructions with two tensor register operands.

    Matches assembly patterns of the form `instr w(vd), m(vs1)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: WeightBuffer
    vs1: MatrixReg
    params: ClassVar[list[Named]] = [w_vd, m_vs1]

@dataclass()
class MXUAccumulatorPush(InstructionPattern):
    """
    Instruction pattern for MXU data push instructions with two tensor register operands.

    Matches assembly patterns of the form `instr a(vd), m(vs1)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: Accumulator
    vs1: MatrixReg
    params: ClassVar[list[Named]] = [acc_vd, m_vs1]

@dataclass()
class MXUAccumulatorPopE1(InstructionPattern):
    """
    Instruction pattern for MXU data pop instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), a(vs2), e(es1)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: MatrixReg
    es1: ExponentReg
    vs2: Accumulator
    params: ClassVar[list[Named]] = [m_vd, acc_vs2, e_es1]

@dataclass()
class MXUAccumulatorPop(InstructionPattern):
    """
    Instruction pattern for MXU data pop instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), a(vs2)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs2: The source tensor register.
    """

    vd: MatrixReg
    vs2: Accumulator
    params: ClassVar[list[Named]] = [m_vd, acc_vs2]

@dataclass()
class MXUMatMul(InstructionPattern):
    """
    Instruction pattern for MXU matrix multiplication.

    Matches assembly patterns of the form `instr a(vd), m(vs1), w(vs2)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination accumulator buffer.
        vs1: The source matrix.
        vs2: The source weights
    """
    vd: Accumulator
    vs1: MatrixReg
    vs2: WeightBuffer
    params: ClassVar[list[Named]] = [acc_vd, m_vs1, w_vs2]

@dataclass()
class DMARegUnary(InstructionPattern):
    """
    Instruction pattern for DMA instructions with one scalar register operand.

    Matches assembly patterns of the form `instr x(rs1)`. This pattern is
    used for `dma.config`.

    Attributes:
        rs1: The source scalar register.
    """

    rs1: ScalarReg
    params: ClassVar[list[Named]] = [x_rs1]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rs1 = ScalarReg(tokens[1])

        return cls(rs1=rs1)

@dataclass()
class Nullary(InstructionPattern):
    """
    Instruction pattern for nullary instructions with no operands.

    Matches assembly patterns of the form `instr`. This pattern is
    used for instructions with no arguments.
    """
    params: ClassVar[list[Named]] = []

@dataclass(init=False)
class UnaryImm(InstructionPattern):
    """
    Instruction pattern for instructions with one immediate operand.

    Matches assembly patterns of the form `instr imm`. This pattern is
    used for `delay`.

    Attributes:
        imm: A 12-bit immediate
    """
    imm: Imm12
    params: ClassVar[list[Named]] = [imm12]
    
    def __init__(self, imm: int):
        self.imm = Imm12(imm)
