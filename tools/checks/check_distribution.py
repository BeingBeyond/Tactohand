"""Build and exercise the selected distribution outside the research checkout.

Uses the current interpreter's dependencies, not a clean dependency environment.
No data, MANO assets, downloads, trained checkpoints or GPUs are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def check_document_links(files: list[str]) -> None:
    selected = set(files)
    missing = set()
    for name in files:
        if not name.endswith(".md"):
            continue
        for link in re.findall(r"\]\(([^\s)]+)\)", (ROOT / name).read_text()):
            target = urlsplit(link)
            if target.scheme or target.netloc or not target.path:
                continue
            path = (ROOT / name).parent / unquote(target.path)
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                missing.add(f"{name} -> {link} (outside source root)")
                continue
            if relative not in selected:
                missing.add(f"{name} -> {relative}")
    if missing:
        raise ValueError("Source documentation has missing links:\n" + "\n".join(sorted(missing)))


def wheel_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if not name.endswith("/") and ".dist-info/" not in name
        }


# Run with -I, an unrelated cwd and only the installed target on the module path.
# Loading console entrypoints from metadata also verifies the packaged declarations.
ENTRYPOINT = """
import importlib.metadata
from pathlib import Path
import sys
target = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(target))
name = sys.argv.pop(1)
sys.argv[0] = name
entry = next(ep for ep in importlib.metadata.distribution('tactohand').entry_points
             if ep.group == 'console_scripts' and ep.name == name)
entry.load()()
"""


PROBE = """
from pathlib import Path
import sys
target, work = map(Path, sys.argv[1:3])
sys.path.insert(0, str(target))
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(41)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import training.configs
from training.tactohand_wrench.nn.network import QueryForceWrenchNetwork
from training.tactohand_wrench.runtime.checkpoints import load_network
from core.data.resources import RESOURCE_ROOT
with initialize_config_dir(config_dir=str(target / 'training/conf'), version_base='1.3'):
    cfg = compose(config_name='config', overrides=[
        'experiment=tactohand', 'data=tactohand', 'data.image_size=32',
        'model.wrench.backbone=vit_tiny_patch16_224',
        'model.wrench.pretrained=false', 'model.wrench.pretrained_path=null',
        'model.wrench.temporal_layers=1', 'model.wrench.temporal_heads=3',
        'model.wrench.query_layers=1', 'model.wrench.query_heads=3',
    ])
assert RESOURCE_ROOT.is_relative_to(target)
for name in ('contact_statistics_v5.npz', 'force_statistics_v5.npz'):
    assert (RESOURCE_ROOT / name).is_file(), name
network = QueryForceWrenchNetwork(OmegaConf.to_container(cfg.model.wrench, resolve=True)).eval()
with torch.no_grad():
    network.query_head.output.weight.normal_(std=0.01)
    network.query_head.output.bias.fill_(0.1)
images = np.random.default_rng(41).integers(0, 256, (1, 2, 3, 32, 32), dtype=np.uint8)
tensor = torch.from_numpy(images)
checkpoint = work / 'synthetic.pt'
torch.save(network.state_dict(), checkpoint)
np.savez(work / 'input.npz', images=images)
OmegaConf.save(cfg, work / 'model.yaml', resolve=True)
from training.tactohand_wrench.runtime.inference import decode_predictions
with torch.inference_mode():
    expected = network(tensor)
    rebuilt = load_network(cfg, checkpoint)
    actual = rebuilt(tensor)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    decoded = decode_predictions(actual, cfg)
np.savez(work / 'expected.npz', **{key: value.numpy() for key, value in decoded.items()})
for name, module in list(sys.modules.items()):
    if name.split('.')[0] in {'core', 'training', 'scripts'} and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(target), (name, module.__file__)
print('Installed resources, tiny network and strict checkpoint roundtrip passed')
"""


VERIFY_OUTPUT = """
from pathlib import Path
import sys
import numpy as np
work = Path(sys.argv[1])
assert (work / 'prediction.done').is_file()
assert (work / 'prediction.json').is_file()
with np.load(work / 'expected.npz') as expected, np.load(work / 'prediction.npz') as actual:
    assert set(expected.files).issubset(actual.files)
    for name in expected.files:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual['contact_prob'].shape == (1, 2, 2, 778)
    assert np.isfinite(actual['force_normalized_10hz']).all()
print('Installed inference CLI matches direct network decoding')
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    output, logs = args.output.resolve(), args.log_dir.resolve()
    done = output.with_suffix(".done")
    output.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    done.unlink(missing_ok=True)
    report: dict[str, Any] = {"status": "running", "python": sys.executable,
              "dependency_environment": "existing dependencies reused; not a clean install",
              "checks": []}
    checks: list[str] = report["checks"]
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "RESUME_FROM", "IMAGE_SIZE"}}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1")

    def run(name: str, command: list[str], cwd: Path) -> None:
        with (logs / f"{name}.log").open("w") as stream:
            subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
        checks.append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {name}", flush=True)

    try:
        scope = runpy.run_path(str(ROOT / "tools/packaging/public_source_scope.py"))
        files = scope["source_files"]()
        check_document_links(files)
        checks.append("source-document-links")
        report["source_files"] = len(files)
        with tempfile.TemporaryDirectory(prefix="tactohand-distribution-") as temporary:
            base = Path(temporary)
            run("source-zip", [sys.executable, str(ROOT / "tools/packaging/build_public_source.py"),
                               "--output", str(base / "source.zip")], base)
            with zipfile.ZipFile(base / "source.zip") as archive:
                archive.extractall(base / "source")
            source = base / "source/Tactohand"
            launcher_bin = base / "bin"
            launcher_bin.mkdir()
            (launcher_bin / "release-python").symlink_to(sys.executable)
            launcher_env = dict(env, PYTHON_BIN="release-python", GPUS="1",
                                PATH=str(launcher_bin) + os.pathsep + env.get("PATH", ""),
                                LOG_DIR=str(logs / "launcher"), V2C_RUN_VERSION="distribution-smoke")
            with (logs / "source-launcher.log").open("w") as stream:
                subprocess.run(["bash", "scripts/training/run_experiment.sh", "tactohand",
                                "--data", "tactohand", "--log-dir", str(logs / "launcher"),
                                "--dry-run"],
                               cwd=source, env=launcher_env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, timeout=180)
            launcher_config = logs / "launcher/distribution-smoke/resolved_config.yaml"
            if not launcher_config.is_file() or not launcher_config.stat().st_size:
                raise ValueError("Source launcher failed to export configuration; see launcher logs")
            run("source-launcher-config", [sys.executable, "-I", "-c",
                "from omegaconf import OmegaConf; import sys; cfg=OmegaConf.load(sys.argv[1]); "
                "assert cfg.task == 'tactohand_wrench'; assert int(cfg.data.image_size) == 256", str(launcher_config)], base)
            run("source-wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps",
                                 "--no-build-isolation", "--wheel-dir", str(base / "wheels"), "."], source)
            wheel = next((base / "wheels").glob("*.whl"))
            payload = wheel_payload(wheel)
            expected = {name for name in files if scope["allows_wheel"](name)}
            if set(payload) != expected:
                raise ValueError(f"Wheel scope mismatch: missing={sorted(expected - set(payload))}; "
                                 f"unexpected={sorted(set(payload) - expected)}")
            checks.append("wheel-file-scope")
            report["wheel_files"] = len(payload)
            run("sdist", [sys.executable, "setup.py", "sdist", "--dist-dir", str(base / "sdist")], source)
            with tarfile.open(next((base / "sdist").glob("*.tar.gz"))) as archive:
                archive.extractall(base / "unpacked", filter="data")
            unpacked = next((base / "unpacked").iterdir())
            run("sdist-wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps",
                                "--no-build-isolation", "--wheel-dir", str(base / "rebuilt"), "."], unpacked)
            if wheel_payload(next((base / "rebuilt").glob("*.whl"))) != payload:
                raise ValueError("Source ZIP and sdist produce different library payloads")
            checks.append("sdist-wheel-payload-parity")
            target = base / "installed"
            run("install", [sys.executable, "-m", "pip", "install", "--no-deps",
                            "--target", str(target), str(wheel)], base)
            work = base / "consumer"
            work.mkdir()
            for name, options in (
                ("tactohand-infer", ["--help"]),
                ("tactohand-view", ["--help"]),
                ("tactohand-eval", ["--help"]),
                ("tactohand-eval", ["--list-metrics"]),
                ("tactohand-train", ["--cfg", "job", "--resolve"]),
            ):
                run(name + "-" + options[0].removeprefix("--"),
                    [sys.executable, "-I", "-c", ENTRYPOINT, str(target), name, *options], work)
            run("installed-network", [sys.executable, "-I", "-c", PROBE, str(target), str(work)], work)
            run("installed-inference", [sys.executable, "-I", "-c", ENTRYPOINT, str(target),
                "tactohand-infer", "--config", str(work / "model.yaml"), "--ckpt", str(work / "synthetic.pt"),
                "--input", str(work / "input.npz"), "--output", str(work / "prediction.npz")], work)
            run("prediction-parity", [sys.executable, "-I", "-c", VERIFY_OUTPUT, str(work)], work)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"FAIL: {error}; see {logs}", file=sys.stderr)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "passed":
        done.touch()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
