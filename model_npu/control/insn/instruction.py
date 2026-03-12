from types import NoneType
from typing import Callable
import inspect
import ast
from model_npu.control.passes import register, mem, functional_unit

# Global registry accumulating decode table rows
_decode_table: list[dict] = []

# Column order — matches the signal list in AtlasCtrlSigs.decode()
_COL_HEADERS = [
    "valid",
    "br_type",
    "src1",
    "src2",
    "dst",
    "msrc1",
    "msrc2",
    "mdst",
    "mem_read",
    "mem_write",
    "wb",
    "pc",
    "mxu_0_valid",
    "mxu_1_valid",
    "scalar_valid",
    "vpu_valid",
    "xlu_valid",
    "dma_valid",
    "alu_op",
    "vpu_op",
    "xlu_op",
]

# Default row — emitted by AtlasCtrlSigs.default
# Analogous to IntCtrlSigs.default in Rocket
_DEFAULT = {
    "valid": "N",
    "br_type": "BR_X",
    "src1": "X",
    "src2": "X",
    "dst": "X",
    "msrc1": "X",
    "msrc2": "X",
    "mdst": "X",
    "mem_read": "N",
    "mem_write": "N",
    "wb": "X",
    "pc": "X",
    "mxu_0_valid": "N",
    "mxu_1_valid": "N",
    "scalar_valid": "N",
    "vpu_valid": "N",
    "xlu_valid": "N",
    "dma_valid": "N",
    "alu_op": "ALU_OP_X",
    "vpu_op": "VPU_OP_X",
    "xlu_op": "XLU_OP_X",
}


def _generate_chisel_table():
    if not _decode_table:
        return

    # Compute column widths
    name_width = max(len(r["name"]) for r in _decode_table)
    col_widths = {h: len(h) for h in _COL_HEADERS}
    for row in _decode_table:
        for h in _COL_HEADERS:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    def fmt_row(name, values: dict, arrow="->") -> str:
        name_part = f"    {name:<{name_width}}"
        vals = ", ".join(str(values[h]).ljust(col_widths[h]) for h in _COL_HEADERS)
        return f"{name_part} {arrow} List({vals})"

    default_vals = ", ".join(_DEFAULT[h].ljust(col_widths[h]) for h in _COL_HEADERS)
    header_vals = {h: h for h in _COL_HEADERS}

    # Emit full Chisel class, mirroring Rocket's IntCtrlSigs + IDecode
    print("class AtlasCtrlSigs(implicit val p: Parameters) extends Bundle {")
    print("  // ... signal declarations ...")
    print()
    print("  def default: List[BitPat] =")
    print(f"    //         {'  '.join(h for h in _COL_HEADERS)}")
    print(f"    List({default_vals})")
    print()
    print("  def decode(inst: UInt, table: Iterable[(BitPat, List[BitPat])]) = {")
    print("    val decoder = DecodeLogic(inst, default, table)")
    print(f"    val sigs = Seq({', '.join(_COL_HEADERS)})")
    print("    sigs zip decoder map { case (s, d) => s := d }")
    print("    this")
    print("  }")
    print("}")
    print()
    print("class AtlasDecode(implicit val p: Parameters) extends DecodeConstants {")
    print("  // Auto-generated decode table")
    print("  // " + fmt_row("instruction", header_vals, arrow="  ").lstrip())
    print("  val table: Array[(BitPat, List[BitPat])] = Array(")

    rows = []
    for row in _decode_table:
        rows.append(fmt_row(row["name"], row))
    print(",\n".join(rows))
    print("  )")
    print("}")


def instr(name=None, *, instruction_type=None):
    def decorator(fn):
        instr_name = (fn.__name__).upper()

        src = inspect.getsource(fn)
        tree = ast.parse(src)

        # Valid
        valid = register.is_valid_instruction(tree)

        # BR_TYPE
        br_type = register.test_conditional_assignment(tree)

        # Register reads/writes
        src1 = register.scalar_src1(tree)
        src2 = register.scalar_src2(tree)
        rd = register.has_register_write_rd(tree)
        mrs1 = register.matrix_src1(tree)
        mrs2 = register.matrix_src2(tree)
        mrd = register.has_register_write_mrd(tree)
        wb = register.has_wb_write(tree)

        # PC update
        pc = register.has_pc_assignment(tree)

        # Functional unit
        category, subtype = functional_unit.instruction_type(tree)
        mxu_0_valid = mxu_1_valid = scalar_valid = False
        vpu_valid = dma_valid = xlu_valid = False

        match category:
            case "MATRIX":
                match subtype:
                    case "MATRIX_SYSTOLIC":
                        mxu_0_valid = True
                    case "MATRIX_IPT":
                        mxu_1_valid = True
            case "SCALAR":
                scalar_valid = True
            case "VECTOR":
                vpu_valid = True
            case "DMA":
                dma_valid = True
            case "TRANSPOSE":
                xlu_valid = True

        # ALU op
        alu_op = functional_unit.scalar_alu_op(tree)
        alu_op_str = (
            "ALU_OP_X" if alu_op is None else f"ALU_OP_{type(alu_op).__name__.upper()}"
        )

        # VPU op
        vpu_op = functional_unit.vector_op(tree)
        if not (type(vpu_op) == str) and vpu_op != None:
            vpu_op = type(vpu_op).__name__

        vpu_op_str = "VPU_OP_X" if vpu_op is None else f"VPU_OP_{vpu_op.upper()}"

        # XLU op
        xlu_op = functional_unit.transpose_op(tree)
        if not (type(xlu_op) == str) and xlu_op != None:
            xlu_op = type(xlu_op).__name__

        xlu_op_str = "XLU_OP_X" if xlu_op is None else f"XLU_OP_{xlu_op.upper()}"

        # mem read/write
        mem_read = mem.check_mem_read(tree)
        mem_write = mem.check_mem_write(tree)

        def _bool(v) -> str:
            if v is True:
                return "Y"
            if v is False:
                return "N"
            return str(v) if v is not None else "X"

        _decode_table.append(
            {
                "name": instr_name,
                "valid": _bool(valid),
                "br_type": str(br_type) if br_type is not None else "BR_X",
                "src1": str(src1),
                "src2": str(src2),
                "dst": _bool(rd),
                "msrc1": mrs1,
                "msrc2": mrs2,
                "mdst": _bool(mrd),
                "mem_read": _bool(mem_read),
                "mem_write": _bool(mem_write),
                "wb": _bool(wb),
                "pc": _bool(pc),
                "mxu_0_valid": _bool(mxu_0_valid),
                "mxu_1_valid": _bool(mxu_1_valid),
                "scalar_valid": _bool(scalar_valid),
                "vpu_valid": _bool(vpu_valid),
                "xlu_valid": _bool(xlu_valid),
                "dma_valid": _bool(dma_valid),
                "alu_op": alu_op_str,
                "vpu_op": vpu_op_str,
                "xlu_op": xlu_op_str,
            }
        )

        return fn

    return decorator
