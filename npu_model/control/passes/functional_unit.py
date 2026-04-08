import ast
from sympy import comp


def parse_instruction_type(type_str: str):
    parts = type_str.split(".")  # ["InstructionType", "SCALAR", "R"]

    match parts:
        case ["InstructionType", category]:
            # e.g. InstructionType.VECTOR, InstructionType.DMA
            return category, None
        case ["InstructionType", category, subtype]:
            # e.g. InstructionType.SCALAR.R
            return category, subtype
        case _:
            raise ValueError(f"Unexpected instruction type: {type_str}")


# which functional unit does this instruction need?
def instruction_type(tree: ast.Module):
    func_def = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )

    for decorator in func_def.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "instruction_type":
                    instr_type = ast.unparse(keyword.value)
                    return parse_instruction_type(instr_type)

    return None


def scalar_alu_op(tree: ast.Module) -> ast.operator():
    category, subtype = instruction_type(tree)
    match category:
        case "SCALAR":
            comp_uses = [
                node for node in ast.walk(tree) if isinstance(node, ast.Compare)
            ]

            for comp_use in comp_uses:
                match comp_use:
                    case ast.Compare(ops=[cmp_op]):
                        return cmp_op
            op_uses = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
            for op_use in op_uses:
                match op_use:
                    case ast.BinOp(op=operator):
                        return operator

    return None


def vector_op(tree: ast.Module):
    category, subtype = instruction_type(tree)
    match category:
        case "VECTOR":
            op_uses = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
            for op_use in op_uses:
                match op_use:
                    case ast.BinOp(op=operator):
                        return operator
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func

                    if isinstance(func, ast.Attribute) and "write_mrf" in func.attr:
                        for arg in node.args:
                            if isinstance(arg, ast.Call):
                                current_node = arg
                                deepest_call = arg

                                while isinstance(current_node, ast.Call) and isinstance(
                                    current_node.func, ast.Attribute
                                ):
                                    deepest_call = current_node
                                    current_node = current_node.func.value

                                if isinstance(deepest_call.func, ast.Attribute):
                                    return deepest_call.func.attr
                                else:
                                    return ast.unparse(deepest_call.func)
    return None


def transpose_op(tree: ast.Module):
    category, subtype = instruction_type(tree)
    match category:
        case "TRANSPOSE":
            match subtype:
                case "L":
                    return "TRANSPOSE_L"
                case "H":
                    return "TRANSPOSE_H"
    return None
