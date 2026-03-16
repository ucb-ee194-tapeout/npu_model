import ast


def _find_mem_subscript(tree: ast.Module) -> ast.Subscript | None:
    """Returns the single state.mem[...] subscript node, or None."""
    found = []
    for node in ast.walk(tree):
        match node:
            case ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="state"),
                    attr="mem",
                ),
            ):
                found.append(node)
    assert len(found) <= 1, f"Expected at most one state.mem access, found {len(found)}"
    return found[0] if found else None


def check_mem(tree: ast.Module) -> bool:
    """Returns True if the instruction accesses state.mem."""
    return _find_mem_subscript(tree) is not None


def check_mem_read(tree: ast.Module) -> bool:
    """Returns True if the state.mem access is a read (Load context)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and "read_mem" in func.id:
                return True

            if isinstance(func, ast.Attribute) and "read_mem" in func.attr:
                return True

    return False


def check_mem_write(tree: ast.Module) -> bool:
    """Returns True if the state.mem access is a write (Store context)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and "write_mem" in func.id:

                return True

            if isinstance(func, ast.Attribute) and "write_mem" in func.attr:
                return True

    return False


def get_mem_base(tree: ast.Module) -> str | int:
    """
    Returns the base of the DMA slice state.mem[base : base + size].
    base is a param name (str) or integer literal (int).
    """
    node = _find_mem_subscript(tree)
    assert node is not None, "No state.mem access found"
    match node.slice:
        case ast.Slice(lower=ast.Name(id=name)):
            return name
        case ast.Slice(lower=ast.Constant(value=val)):
            return val
    raise AssertionError(f"Could not extract base from slice: {ast.dump(node.slice)}")


def get_mem_size(tree: ast.Module) -> str | int:
    """
    Returns the size of the DMA slice state.mem[base : base + size].
    size is a param name (str) or integer literal (int).
    """
    node = _find_mem_subscript(tree)
    assert node is not None, "No state.mem access found"
    match node.slice:
        case ast.Slice(
            upper=ast.BinOp(
                op=ast.Add(),
                right=ast.Name(id=name),
            )
        ):
            return name
        case ast.Slice(
            upper=ast.BinOp(
                op=ast.Add(),
                right=ast.Constant(value=val),
            )
        ):
            return val
    raise AssertionError(f"Could not extract size from slice: {ast.dump(node.slice)}")
