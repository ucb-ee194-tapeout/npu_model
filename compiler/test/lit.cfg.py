import os
import shutil

import lit.formats

config.name = "NPU"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "outputs"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = getattr(config, "test_exec_root", config.test_source_root)

config.llvm_tools_dir = getattr(config, "llvm_tools_dir", "")
config.npu_tools_dir = getattr(config, "npu_tools_dir", "")

path_entries = [p for p in [config.npu_tools_dir, config.llvm_tools_dir] if p]
if path_entries:
    old_path = config.environment.get("PATH", "")
    config.environment["PATH"] = os.pathsep.join(path_entries + [old_path])

for tool in ["npu-opt", "npu-translate", "FileCheck"]:
    if shutil.which(tool, path=config.environment.get("PATH", "")) is None:
        lit_config.fatal(f"required test tool not found on PATH: {tool}")
