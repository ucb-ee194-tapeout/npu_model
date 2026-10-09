#!/usr/bin/env python3
"""Regenerate checked-in traces and exhaustive unary tables with Chisel/Verilator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

MODEL = Path(__file__).resolve().parents[1]
RTL = MODEL.parent
CHIPYARD = RTL.parents[1]
MANIFEST = MODEL / 'tests/rtl/provenance.json'
# Harnesses for src/main/scala/atlas, run by the accelerator's Mill module.
SUITES = ['atlas.scalar.NpuModelScalarTraceTest',
          'atlas.vector.NpuModelVectorTraceTest',
          'atlas.vector.NpuModelUnaryTableTest',
          'atlas.mxu.NpuModelMatrixTraceTest',
          'atlas.mxu.NpuModelArithmeticTraceTest',
          'atlas.lsu.NpuModelMemoryTraceTest']
# Harnesses for src/main/scala/diplomatic (DMA, VMEM), which need rocket-chip
# and are compiled only by Chipyard's sbt project `sp26atlas`.
SBT_HARNESS_DIR = 'dma'
SBT_SUITES = ['atlas.dma.NpuModelDmaTraceTest']
SBT_OPTS = ['-Dsbt.ivy.home={cy}/.ivy2', '-Dsbt.global.base={cy}/.sbt',
            '-Dsbt.boot.directory={cy}/.sbt/boot/', '-Dsbt.supershell=false',
            '-Dsbt.server.forcestart=true']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    roots = [RTL/'src/main/scala', RTL/'dependencies/sp26-fp-units/src/main/scala',
             RTL/'dependencies/fpex/src/main/scala', MODEL/'tests/rtl/scala']
    return sorted(p for root in roots for p in root.rglob('*.scala'))


def artifact_paths():
    return sorted([p for p in (MODEL/'tests/rtl').glob('*.json') if p != MANIFEST]
                  + list((MODEL/'npu_model/configs/data').glob('*.bin.gz')))


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
    parser.add_argument('--only',metavar='TEXT',help='Run only suites whose name contains TEXT (manifest still covers everything)')
    args=parser.parse_args()
    if args.check:
        check_manifest()
        print('RTL source and artifact fingerprints match')
        return
    selected = lambda suite: args.only is None or args.only in suite
    mill_suites = [s for s in SUITES if selected(s)]
    sbt_suites = [s for s in SBT_SUITES if selected(s)]
    if mill_suites:
        # Mill includes this test-source tree. Keep the canonical harnesses in the
        # model repo, and install temporary copies only for the simulator invocation.
        with tempfile.TemporaryDirectory(prefix='npu_model_',dir=RTL/'src/test/scala/atlas') as work:
            shutil.copytree(MODEL/'tests/rtl/scala',Path(work)/'harnesses',
                            ignore=shutil.ignore_patterns(SBT_HARNESS_DIR))
            subprocess.run([str(RTL/'mill'),'--no-server','atlas.test.testOnly',*mill_suites],cwd=RTL,check=True)
    if sbt_suites:
        # sbt excludes src/test/scala/atlas (the Mill tree), so install beside it.
        with tempfile.TemporaryDirectory(prefix='npu_model_',dir=RTL/'src/test/scala') as work:
            shutil.copytree(MODEL/'tests/rtl/scala'/SBT_HARNESS_DIR,Path(work)/'harnesses')
            env = {**os.environ, 'NPU_MODEL_ROOT': str(MODEL)}
            env.setdefault('JAVA_TOOL_OPTIONS', f'-Xmx8G -Xss8M -Djava.io.tmpdir={CHIPYARD}/.java_tmp')
            subprocess.run(['java','-jar',str(CHIPYARD/'scripts/sbt-launch.jar'),
                            *[opt.format(cy=CHIPYARD) for opt in SBT_OPTS],
                            'project sp26atlas',f'testOnly {" ".join(sbt_suites)}'],
                           cwd=CHIPYARD,env=env,check=True)
    write_manifest()
    check_manifest()


if __name__ == '__main__':
    main()
