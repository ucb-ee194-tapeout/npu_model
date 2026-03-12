import ast
from model_npu.control.passes.functional_unit import instruction_type


def extract_assignments(tree: ast.Module):
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments.append(node)
    return assignments


def extract_conditionals(tree: ast.Module):
    conditionals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            conditionals.append(node)
    return conditionals


def extract_subscripts(tree: ast.Module):
    subscripts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            subscripts.append(node)
    return subscripts


def is_valid_instruction(node: ast.AST):
    if node is not None:
        return "Y"
    else:
        return "N"


# is there a register assignment in the body?
def has_register_write_rd(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Name) and func.id == "write_xrf":
                return True

            if isinstance(func, ast.Attribute) and func.attr == "write_xrf":
                return True

    return False


def scalar_src1(tree: ast.Module):
    category, subtype = instruction_type(tree)
    result = "OP1_X"
    match category:
        case "SCALAR":
            match subtype:
                case "R":
                    result = "OP1_RS1"
                case "I":
                    result = "OP1_RS1"  # FIXME - do we have JALR?
                case "J":
                    result = "OP1_PC"
                case "B":
                    result = "OP1_RS1"
                case "U":
                    result = "OP1_IMM"  # FIXME - do we have AUIPC?
    return result


def scalar_src2(tree: ast.Module):
    category, subtype = instruction_type(tree)
    result = "OP2_X"
    match category:
        case "SCALAR":
            match subtype:
                case "R":
                    result = "OP2_RS1"
                case "I":
                    result = "OP2_IMM"
                case "J":
                    result = "OP2_IMM"
                case "B":
                    result = "OP2_RS2"
                case "U":
                    result = "OP2_X"
    return result


def has_register_write_mrd(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and "write_mrf" in func.id:
                return True

            if isinstance(func, ast.Attribute) and "write_mrf" in func.attr:
                return True

    return False


def matrix_src1(tree: ast.Module):
    category, subtype = instruction_type(tree)
    result = "OP1_X"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not (isinstance(func, ast.Name) and "read_mrf" in func.id):
                for i, arg in enumerate(node.args):
                    if "rs1" in ast.unparse(arg):
                        result = "OP1_MRS1"

            if not (isinstance(func, ast.Attribute) and "read_mrf" in func.attr):
                for i, arg in enumerate(node.args):
                    if "mrs1" in ast.unparse(arg):
                        result = "OP1_MRS1"

    return result


def matrix_src2(tree: ast.Module):
    category, subtype = instruction_type(tree)
    result = "OP2_X"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Name)
                and ("read_mrf" in func.id or "read_wb" in func.id)
            ):
                for i, arg in enumerate(node.args):
                    if "rs2" in ast.unparse(arg):
                        result = "OP2_MRS2"

            if not (
                isinstance(func, ast.Attribute)
                and ("read_mrf" in func.attr or "read_wb" in func.attr)
            ):
                for i, arg in enumerate(node.args):
                    if "rs2" in ast.unparse(arg):
                        result = "OP2_MRS2"

    return result


# is there a weight buffer assignment in the body?
def has_wb_write(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and "write_wb" in func.id:

                return True

            if isinstance(func, ast.Attribute) and "write_wb" in func.attr:
                return True

    return False


def test_register_assignment(tree: ast.Module):
    assignments = extract_assignments(tree)
    for assignment in assignments:
        if (
            len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Subscript)
            and isinstance(assignment.targets[0].value, ast.Name)
            and assignment.targets[0].value.id == "regfile"
        ):
            return True
    return False


def has_pc_assignment(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and "set_npc" in func.id:

                return True

            if isinstance(func, ast.Attribute) and "set_npc" in func.attr:
                return True

    return False


def test_conditional_assignment(tree: ast.Module):
    conditionals = extract_conditionals(tree)
    for conditional in conditionals:
        if len(conditional.test.ops) != 1:
            continue
        if isinstance(conditional.test.ops[0], ast.Eq):
            return "BR_EQ"
        elif isinstance(conditional.test.ops[0], ast.NotEq):
            return "BR_NE"
        elif isinstance(conditional.test.ops[0], ast.Lt):
            return "BR_LT"
        elif isinstance(conditional.test.ops[0], ast.LtE):
            return "BR_LE"
        elif isinstance(conditional.test.ops[0], ast.Gt):
            return "BR_GT"
    return "BR_N"
