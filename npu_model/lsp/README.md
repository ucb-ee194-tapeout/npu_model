# NPU Assembly LSP Linter

A Language Server Protocol (LSP) linter for NPU assembly (`.S`) files, integrated with VSCode. It validates instructions against the NPU ISA defined in `npu_model` and reports errors and warnings inline as you type.

## What it checks

- **Unknown mnemonics** — flags any opcode not in `IsaSpec.operations`
- **Wrong operand count** — tells you exactly how many operands an instruction expects
- **Invalid registers** — wrong prefix (`x`/`e`/`m`/`a`/`w`) or out-of-range index
- **Immediates out of range** — validated against the correct bit-width for each instruction type (Imm12, Imm16, Imm20, shamt)
- **Malformed base+offset operands** — e.g. `lw x1, bad` instead of `lw x1, 16(x2)`
- **Odd branch offsets** — SB-type branches require even offsets
- **Duplicate labels** — warns when the same label is defined twice
- **Malformed label names** — must match `[A-Za-z_][A-Za-z0-9_.]*`

It also provides **mnemonic autocomplete** for all real and pseudo instructions.

---

## File layout

```
npu_model/lsp/
├── linter.py                      # Core linting logic (no LSP dependency)
├── server.py                      # pygls LSP server
└── vscode-extension/
    ├── package.json               # Extension manifest
    ├── language-configuration.json
    ├── src/
    │   └── extension.js           # Extension entry point (starts server)
    └── syntaxes/
        └── npu-asm.tmLanguage.json  # Syntax highlighting grammar
```

---

## Requirements

| Tool | Purpose |
|------|---------|
| `uv` | Runs the Python server and installs `pygls`/`lsprotocol` on-the-fly |
| VSCode ≥ 1.75 | Hosts the extension |
| Node / npm | Only needed if you want to `vsce package` the extension |

The Python server imports `npu_model` from the project root, so **no separate install step is needed** — `uv run` picks up the project's existing virtual environment (which already has `torch`, `npu_model`, etc.).

---

## Setup

### 1 — Install Node dependencies (first time only)

From the **project root**:

```bash
cd npu_model/lsp/vscode-extension
npm install --save vscode-languageclient
```

### 2 — Symlink the extension into VSCode

From the **project root**:

```bash
ln -sf "$(pwd)/npu_model/lsp/vscode-extension" \
      ~/.vscode/extensions/npu-model.npu-asm-0.1.0
```

The name of the symlink (`npu-model.npu-asm-0.1.0`) must match the `publisher.name-version` format from `package.json` exactly — VSCode uses it to identify the extension.

### 3 — Reload VSCode

Run Developer: Reload Window. The extension will activate automatically the next time you open a `.S`, `.s`, or `.asm` file.

### 5 — Verify `uv` is on your PATH

```bash
uv --version
```

If `uv` is not on your PATH, set the `npu-asm.pythonPath` setting (see below) to point at a Python interpreter that already has `pygls` and `lsprotocol` installed.

---

## Configuration

All settings live under the `npu-asm` namespace in VSCode settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `npu-asm.pythonPath` | `""` | Explicit Python interpreter path. When empty the extension uses `uv run` from the project root (recommended). |
| `npu-asm.serverArgs` | `[]` | Extra arguments forwarded verbatim to `server.py`. |

Example `settings.json` if you need an explicit interpreter:

```json
{
  "npu-asm.pythonPath": "/path/to/your/project/.venv/bin/python"
}
```

---

## How the server is launched

When a `.S` / `.s` / `.asm` file is opened, the extension runs:

```bash
uv run --with pygls --with lsprotocol npu_model/lsp/server.py
```

from the **project root** (the directory containing `pyproject.toml`). This means:

- `pygls` and `lsprotocol` are injected into the project's existing venv automatically.
- `npu_model`, `torch`, and all other project dependencies are already available.
- No manual `pip install` is ever required.

The server communicates with VSCode over **stdio** using the Language Server Protocol.

---

## Restarting the server

Run the command palette action **NPU Assembly: Restart Server** (`npu-asm.restartServer`) to bounce the language server without reloading VSCode. This is useful after editing `linter.py` or `server.py`.

---

## Running the linter standalone

You can lint a file from the terminal without VSCode:

```bash
uv run python -c "
import sys
sys.path.insert(0, '.')
from npu_model.lsp.linter import lint_text, Severity
for d in lint_text(open('my_program.S').read()):
    print(f'  line {d.line+1}: {d.message}')
"
```

Or import `lint_text` directly in a test:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))  # project root
from npu_model.lsp.linter import lint_text, Severity

diags = lint_text(open("bad.S").read())
errors = [d for d in diags if d.severity == Severity.ERROR]
```

---

## Extending the linter

All linting logic lives in `linter.py`. To add a new check:

1. Add or modify a `lint()` classmethod on the relevant `InstructionPattern` subclass in `isa_patterns.py`, or add a pseudo-instruction handler in `linter.py`.
2. Run **NPU Assembly: Restart Server** from the command palette to pick up the change without restarting VSCode.