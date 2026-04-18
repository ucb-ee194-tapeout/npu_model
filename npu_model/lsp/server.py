"""
NPU Assembly LSP Server

Launch via the VSCode extension (which calls `uv run --with pygls --with
lsprotocol server.py` from the project root), or manually:

    uv run --with pygls --with lsprotocol \
        "npu_model/npu_model/lsp/server.py"
"""

import logging
import re
import sys
from pathlib import Path
from typing import cast

# ---------------------------------------------------------------------------
# Bootstrap sys.path so `import npu_model` works.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Also make linter.py importable (it lives next to this file).
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from linter import Severity, SEEN_LABELS, lint_text
from lsprotocol import types
from pygls.lsp.server import LanguageServer
from npu_model.isa_patterns import InstructionPattern, x_rd
from npu_model.isa_types import Named, Bundled, Imm32

# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------

_PSEUDO_PARAMS: dict[str, list[Named]] = {
    "nop": [],
    "li": [x_rd, Named(Imm32, "32-bit immediate", "imm", False)],
}

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NPU_LS_NAME = "npu-asm-lsp"
NPU_LS_VERSION = "0.1.0"

server = LanguageServer(
    NPU_LS_NAME,
    NPU_LS_VERSION,
    text_document_sync_kind=types.TextDocumentSyncKind.Full,
)

# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    Severity.ERROR: types.DiagnosticSeverity.Error,
    Severity.WARNING: types.DiagnosticSeverity.Warning,
    Severity.INFORMATION: types.DiagnosticSeverity.Information,
    Severity.HINT: types.DiagnosticSeverity.Hint,
}


def _publish(ls: LanguageServer, uri: str, source: str) -> None:
    results = lint_text(source)
    lsp_diags: list[types.Diagnostic] = []

    for d in results:
        lsp_diags.append(
            types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=d.line, character=d.col_start),
                    end=types.Position(line=d.line, character=d.col_end),
                ),
                message=d.message,
                severity=_SEVERITY_MAP.get(d.severity, types.DiagnosticSeverity.Error),
                source=NPU_LS_NAME,
            )
        )

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diags)
    )
    log.info("Published %d diagnostic(s) for %s", len(lsp_diags), uri)


# ---------------------------------------------------------------------------
# LSP event handlers
# ---------------------------------------------------------------------------


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    _publish(ls, params.text_document.uri, params.text_document.text)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    if params.content_changes:
        _publish(ls, params.text_document.uri, params.content_changes[-1].text)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    # text is only present when the client sends includeText; fall back to
    # the workspace copy if it isn't.
    text = getattr(params, "text", None)
    if text is None:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        text = doc.source
    _publish(ls, params.text_document.uri, text)


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
    )


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completions(
    ls: LanguageServer, params: types.CompletionParams
) -> types.CompletionList:
    
    doc = ls.workspace.get_text_document(params.text_document.uri)
    line = doc.lines[params.position.line]

    # Strip comment, work only up to the cursor
    text = line[: params.position.character]
    comment = text.find("#")
    if comment != -1:
        text = text[:comment]

    # Tokenise (same rules as the assembler: space/comma separators)
    tokens = [t for t in re.split(r"[\s,]+", text) if t]

    if len(tokens) == 0 or (len(tokens) == 1 and not text.endswith(" ")):
        try:
            from npu_model.isa import IsaSpec
            items: list[types.CompletionItem] = []
            for mnemonic, instr_cls in sorted(IsaSpec.operations.items()):
                instr_cls = cast(InstructionPattern, instr_cls)
                detail = ', '.join(instr_cls.format_params())
                items.append(
                    types.CompletionItem(
                        label=mnemonic,
                        kind=types.CompletionItemKind.Keyword,
                        detail=detail,
                    )
                )

            for pseudo, p in _PSEUDO_PARAMS.items():
                detail = ", ".join(map(lambda p: p.format_arg(), p)) if p else "no operands"
                items.append(
                    types.CompletionItem(
                        label=pseudo,
                        kind=types.CompletionItemKind.Keyword,
                        detail=detail,
                    )
                )
            return types.CompletionList(is_incomplete=False, items=items)
        except Exception:
            return types.CompletionList(is_incomplete=False, items=[])
    else:
        try:
            from npu_model.isa import IsaSpec

            mnemonic = tokens[0].lower()

            if mnemonic in _PSEUDO_PARAMS:
                operand_params = _PSEUDO_PARAMS[mnemonic]
            elif mnemonic in IsaSpec.operations:
                operand_params = cast(InstructionPattern, IsaSpec.operations[mnemonic]).params
            else:
                raise Exception("mnemonic isn't real")
            
            decomposed_params: list[Named] = []
            for param in operand_params:
                if isinstance(param, Bundled):
                    decomposed_params.append(param.imm)
                    decomposed_params.append(param.reg)
                else:
                    decomposed_params.append(param)
            
            # How many operands has the user already started typing?
            # tokens[0] is the mnemonic; each subsequent token is an operand.
            if len(text) != 0 and len(tokens) != 0 and (text[-1] == " " or (isinstance(operand_params[-1], Bundled) and tokens[-1].find('(') != -1)):
                active = max(0, min(len(tokens) - 1, len(decomposed_params) - 1))
            else:
                active = max(0, min(len(tokens) - 2, len(decomposed_params) - 1))
            
            labels = list(SEEN_LABELS.keys())
            items: list[types.CompletionItem] = []
            for item in decomposed_params[active].autocomplete(labels):
                items.append(
                    types.CompletionItem(
                        label=item,
                        kind=types.CompletionItemKind.Variable,
                        detail="",
                    )
                )

            return types.CompletionList(is_incomplete=False, items=items)

        except Exception:
            return types.CompletionList(is_incomplete=False, items=[])


# ---------------------------------------------------------------------------
# Signature help
# ---------------------------------------------------------------------------

@server.feature(
    types.TEXT_DOCUMENT_SIGNATURE_HELP,
    types.SignatureHelpOptions(trigger_characters=[" ", ","]),
)
def signature_help(
    ls: LanguageServer, params: types.SignatureHelpParams
) -> types.SignatureHelp | None:
    try:
        from npu_model.isa import IsaSpec

        doc = ls.workspace.get_text_document(params.text_document.uri)
        line = doc.lines[params.position.line]

        # Strip comment, work only up to the cursor
        text = line[: params.position.character]
        comment = text.find("#")
        if comment != -1:
            text = text[:comment]

        # Tokenise (same rules as the assembler: space/comma separators)
        tokens = [t for t in re.split(r"[\s,]+", text) if t]
        if not tokens:
            return None

        mnemonic = tokens[0].lower()

        if mnemonic in _PSEUDO_PARAMS:
            operand_params = list(map(lambda x: x.format_arg(), _PSEUDO_PARAMS[mnemonic]))
        elif mnemonic in IsaSpec.operations:
            operand_params = list(cast(InstructionPattern, IsaSpec.operations[mnemonic]).format_params())
        else:
            return None

        if not operand_params:
            return None

        # Get pattern information together
        parameters: list[types.ParameterInformation] = []
        for param in operand_params:
            if param.endswith(')'):
                split_params = param.split('(')
                parameters.append(types.ParameterInformation(label=split_params[0]))
                parameters.append(types.ParameterInformation(label=split_params[1][:-1]))
            else:
                parameters.append(types.ParameterInformation(label=param))

        # How many operands has the user already started typing?
        # tokens[0] is the mnemonic; each subsequent token is an operand.
        if len(text) != 0 and len(tokens) != 0 and (text[-1] == " " or (operand_params[-1].endswith(')') and tokens[-1].find('(') != -1)):
            active = max(0, min(len(tokens) - 1, len(parameters) - 1))
        else:
            active = max(0, min(len(tokens) - 2, len(parameters) - 1))
        label = f"{mnemonic} {', '.join(operand_params)}"

        sig = types.SignatureInformation(
            label=label,
            parameters=parameters,
        )
        return types.SignatureHelp(
            signatures=[sig],
            active_signature=0,
            active_parameter=active,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting %s %s over stdio", NPU_LS_NAME, NPU_LS_VERSION)
    server.start_io()
