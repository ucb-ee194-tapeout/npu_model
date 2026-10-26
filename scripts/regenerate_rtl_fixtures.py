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
MANIFEST = MODEL / 'tests/rtl/provenance.json'
# Harnesses for src/main/scala/atlas, run by the accelerator's Mill module, and the
# fixtures (globs in npu-model) that each one records.
SUITES = {'atlas.scalar.NpuModelScalarTraceTest': ['tests/rtl/scalar_cases.json', 'tests/rtl/scalar_traces.json'],
          'atlas.vector.NpuModelVectorTraceTest': ['tests/rtl/vector_traces.json'],
          'atlas.vector.NpuModelUnaryTableTest': ['npu_model/configs/data/*.bin.gz'],
          'atlas.mxu.NpuModelMatrixTraceTest': ['tests/rtl/ipt_traces.json'],
          'atlas.mxu.NpuModelArithmeticTraceTest': ['tests/rtl/arithmetic.json'],
          'atlas.lsu.NpuModelMemoryTraceTest': ['tests/rtl/memory_traces.json']}
# Harnesses for src/main/scala/diplomatic (DMA, VMEM), which need rocket-chip
# and are compiled only by Chipyard's sbt project `sp26atlas`.
SBT_HARNESS_DIR = 'dma'
SBT_SUITES = {'atlas.dma.NpuModelDmaTraceTest': ['tests/rtl/dma_traces.json']}
ALL_SUITES = {**SUITES, **SBT_SUITES}
SBT_OPTS = ['-Dsbt.ivy.home={cy}/.ivy2', '-Dsbt.global.base={cy}/.sbt',
            '-Dsbt.boot.directory={cy}/.sbt/boot/', '-Dsbt.supershell=false',
            '-Dsbt.server.forcestart=true']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    roots = [RTL/'src/main/scala', RTL/'dependencies/sp26-fp-units/src/main/scala',
             RTL/'dependencies/fpex/src/main/scala', MODEL/'tests/rtl/scala']
    return sorted(p for root in roots for p in root.rglob('*.scala'))


def sources_digest():
    files = {str(p.relative_to(RTL)): digest(p) for p in source_paths()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def fixture_digests(suite):
    return {str(p.relative_to(MODEL)): digest(p)
            for pattern in ALL_SUITES[suite] for p in sorted(MODEL.glob(pattern))}


def write_manifest(suites):
    """Record the fixtures of `suites` with the current sources. Keep the records of the other suites."""
    records = json.loads(MANIFEST.read_text()).get('suites', {}) if MANIFEST.exists() else {}
    records = {suite: record for suite, record in records.items() if suite in ALL_SUITES}
    sources = sources_digest()
    records.update({suite: {'fixtures': fixture_digests(suite), 'sources': sources} for suite in suites})
    manifest = {'description': 'Per harness suite: SHA256 of its fixtures, and one SHA256 of the '
                               'RTL and harness sources that recorded them.',
                'suites': records}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


def check_manifest():
    records = json.loads(MANIFEST.read_text()).get('suites', {})
    # The Python model can also be used as a standalone checkout, without the RTL sources.
    sources = sources_digest() if (RTL/'src/main/scala/atlas').exists() else None
    for suite in ALL_SUITES:
        if suite not in records:
            raise RuntimeError(f'RTL fixture provenance: no record of {suite}; regenerate fixtures')
        recorded, actual = records[suite]['fixtures'], fixture_digests(suite)
        for name in sorted(recorded.keys() | actual.keys()):
            if recorded.get(name) != actual.get(name):
                raise RuntimeError(f'RTL fixture provenance mismatch: {MODEL/name}; regenerate fixtures')
        if sources is not None and records[suite]['sources'] != sources:
            raise RuntimeError(f'RTL fixture provenance mismatch: the sources changed after {suite} ran; '
                               'regenerate fixtures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check source/artifact fingerprints without simulating')
    parser.add_argument('--only',metavar='TEXT',help='Run only suites whose name contains TEXT (keep the records of the others)')
    parser.add_argument('--chipyard',metavar='DIR',type=Path,default=Path(os.environ.get('CY_DIR',RTL.parents[1])),
                        help='Chipyard root for the sbt suites; its generators/atlas-npu must be this checkout '
                             '(default: $CY_DIR, else two levels above this checkout)')
    args=parser.parse_args()
    if args.check:
        check_manifest()
        print('RTL source and artifact fingerprints match')
        return
    selected = lambda suite: args.only is None or args.only in suite
    mill_suites = [s for s in SUITES if selected(s)]
    sbt_suites = [s for s in SBT_SUITES if selected(s)]
    if not mill_suites + sbt_suites:
        raise SystemExit(f'No suite name contains {args.only!r} (the match is case-sensitive)')
    chipyard = args.chipyard.resolve()
    # Chipyard's sbt project sp26atlas is generators/atlas-npu.
    if sbt_suites and (chipyard/'generators/atlas-npu').resolve() != RTL:
        raise SystemExit(f'{chipyard}/generators/atlas-npu is not {RTL}; see tests/rtl/README.md')
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
            env.setdefault('JAVA_TOOL_OPTIONS', f'-Xmx8G -Xss8M -Djava.io.tmpdir={chipyard}/.java_tmp')
            subprocess.run(['java','-jar',str(chipyard/'scripts/sbt-launch.jar'),
                            *[opt.format(cy=chipyard) for opt in SBT_OPTS],
                            'project sp26atlas',f'testOnly {" ".join(sbt_suites)}'],
                           cwd=chipyard,env=env,check=True)
    write_manifest(mill_suites + sbt_suites)
    check_manifest()


if __name__ == '__main__':
    main()
