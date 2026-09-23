#!/usr/bin/env python3
"""Regenerate checked-in traces and exhaustive unary tables with Chisel/Verilator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

MODEL = Path(__file__).resolve().parents[1]
RTL = MODEL.parent
MANIFEST = MODEL / 'tests/rtl/provenance.json'
SUITES = ['atlas.scalar.NpuModelScalarTraceTest',
          'atlas.vector.NpuModelVectorTraceTest',
          'atlas.vector.NpuModelUnaryTableTest',
          'atlas.mxu.NpuModelMatrixTraceTest',
          'atlas.mxu.NpuModelArithmeticTraceTest',
          'atlas.lsu.NpuModelMemoryTraceTest']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    roots = [RTL/'src/main/scala', RTL/'dependencies/sp26-fp-units/src/main/scala',
             RTL/'dependencies/fpex/src/main/scala', MODEL/'tests/rtl/scala']
    return sorted(p for root in roots for p in root.rglob('*.scala'))


def artifact_paths():
    return sorted([p for p in (MODEL/'tests/rtl').glob('*.json') if p != MANIFEST]
                  + list((MODEL/'npu_model/hardware/data').glob('*.bin.gz')))


def write_manifest():
    manifest = {'description': 'SHA256 of RTL/harness sources and their generated artifacts.',
                'sources': {str(p.relative_to(RTL)): digest(p) for p in source_paths()},
                'artifacts': {str(p.relative_to(MODEL)): digest(p) for p in artifact_paths()}}
    MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')


def check_manifest():
    manifest = json.loads(MANIFEST.read_text())
    for category,root in [('artifacts',MODEL),('sources',RTL)]:
        for name,expected in manifest[category].items():
            path = root/name
            if category == 'sources' and not (RTL/'src/main/scala/atlas').exists():
                continue  # The Python model can also be used as a standalone checkout.
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f'RTL fixture provenance mismatch: {path}; regenerate fixtures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check source/artifact fingerprints without simulating')
    args=parser.parse_args()
    if args.check:
        check_manifest()
        print('RTL source and artifact fingerprints match')
        return
    # Mill includes this test-source tree. Keep the canonical harnesses in the
    # model repo, and install temporary copies only for the simulator invocation.
    with tempfile.TemporaryDirectory(prefix='npu_model_',dir=RTL/'src/test/scala/atlas') as work:
        shutil.copytree(MODEL/'tests/rtl/scala',Path(work)/'harnesses')
        subprocess.run([str(RTL/'mill'),'--no-server','atlas.test.testOnly',*SUITES],cwd=RTL,check=True)
    write_manifest()
    check_manifest()


if __name__ == '__main__':
    main()
