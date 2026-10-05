"""Exercise the actual batch shell with a simulated ABCI module environment."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def check_gpu_job_environment(tmp_path, has_nvcc):
    repo = tmp_path / "checkout"
    scripts = repo / "scripts/abci"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/abci/0390_run.sh", scripts / "0390_run.sh")
    work = tmp_path / "work area"
    venv = work / "env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text(
        f"export PATH={shlex.quote(str(venv / 'bin'))}:\"$PATH\"\n"
    )
    python = venv / "bin/python"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o755)
    (repo / "local.env").write_text(
        f"export CONREP_WORK_ROOT={shlex.quote(str(work))}\n"
        f"export CONREP_ENV={shlex.quote(str(venv))}\n"
    )
    toolkit = tmp_path / "cuda-12.4.1"
    (toolkit / "bin").mkdir(parents=True)
    if has_nvcc:
        nvcc = toolkit / "bin/nvcc"
        nvcc.write_text("#!/bin/sh\nprintf 'Cuda compilation tools, release 12.4\\n'\n")
        nvcc.chmod(0o755)
    # BASH_ENV supplies a module function as an initialized ABCI login would.
    init = tmp_path / "modules.sh"
    init.write_text(
        "module() {\n"
        "  [[ $1 == load && $2 == cuda/12.4/12.4.1 ]] || return 2\n"
        f"  export CUDA_HOME={shlex.quote(str(toolkit))}\n"
        "}\n"
    )
    (scripts / "0390_launch.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path('launch.json').write_text(json.dumps({'env': dict(os.environ), 'args': sys.argv[1:]}))\n"
    )
    env = dict(os.environ, BASH_ENV=str(init))
    env.pop("CONREP_CUDA_MODULE", None)
    env.pop("CUDA_HOME", None)
    result = subprocess.run(
        ["bash", str(scripts / "0390_run.sh"), "--nproc", "8", "--", "sft"],
        env=env,
        capture_output=True,
        text=True,
    )
    probe = repo / "launch.json"
    if not has_nvcc:
        assert result.returncode != 0
        assert "CUDA compiler not found" in result.stderr
        assert not probe.exists(), "Python must not start with a missing toolkit"
        return
    assert result.returncode == 0, result.stderr
    captured = json.loads(probe.read_text())
    assert captured["args"] == ["--nproc", "8", "--", "sft"]
    assert captured["env"]["CUDA_HOME"] == str(toolkit)
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(venv / "bin")
    for key, relative in (
        ("TORCH_EXTENSIONS_DIR", "cache/torch_extensions"),
        ("TRITON_CACHE_DIR", "cache/triton"),
        ("TMPDIR", "tmp"),
    ):
        assert captured["env"][key] == str(work / relative)
        assert (work / relative).is_dir()
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"


class ABCIEnvironmentTests(unittest.TestCase):
    def test_toolkit_and_venv_reach_python(self):
        with tempfile.TemporaryDirectory() as directory:
            check_gpu_job_environment(Path(directory), has_nvcc=True)

    def test_missing_compiler_stops_before_python(self):
        with tempfile.TemporaryDirectory() as directory:
            check_gpu_job_environment(Path(directory), has_nvcc=False)


if __name__ == "__main__":
    unittest.main()
