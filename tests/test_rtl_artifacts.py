"""Guard against stale simulator fixtures and checked-in assembler images."""
import importlib.util
from pathlib import Path
import struct

from npu_model.util.converter import load_asm

ROOT = Path(__file__).resolve().parents[1]


def test_rtl_source_and_artifact_provenance():
    spec = importlib.util.spec_from_file_location('rtl_fixtures', ROOT/'scripts/regenerate_rtl_fixtures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_manifest()


def test_checked_in_assembly_images_match_sources():
    base = ROOT/'npu_model/configs/programs'
    for source in sorted((base/'asm').glob('*.S')):
        words = [insn.to_bytecode() & 0xffffffff for insn in load_asm(source)]
        expected = struct.pack(f'<{len(words)}I',*words)
        assert (base/'bin'/f'{source.stem}.bin').read_bytes() == expected, source.name
        assert (base/'hex'/f'{source.stem}.hex').read_text() == ''.join(f'{word:08x}\n' for word in words), source.name
