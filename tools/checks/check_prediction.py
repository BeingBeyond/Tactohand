"""CPU contracts for standalone prediction and explicit-config evaluation.

Uses a tiny randomly initialized network, synthetic RGB/geometry/label data,
and real model/metric consumers. No dataset mount or MANO assets are required.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
import tempfile
from threading import Thread
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import training.configs

from training.tactohand_wrench.runtime.checkpoints import load_network, load_prediction_config
from training.tactohand_wrench.runtime.inference import decode_predictions, validate_images
from training.tactohand_wrench.model import SubstepForceModel
from training.evaluation.harness import EvalHarness, resolve_checkpoint
from training.infer_tactohand import main as infer_main, read_video
from training.eval_tactohand import main as eval_main
from training.visualization.inference_viewer import ViewerServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    checks = []
    with tempfile.TemporaryDirectory(prefix="tactohand-prediction-") as temporary:
        base = Path(temporary)
        with initialize_config_dir(config_dir=str(ROOT / "training/conf"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[
                "experiment=tactohand", "data=tactohand",
                "model.wrench.backbone=vit_tiny_patch16_224", "data.image_size=32",
                "model.wrench.pretrained=false", "model.wrench.pretrained_path=null",
                "model.wrench.temporal_layers=1", "model.wrench.temporal_heads=3",
                "model.wrench.query_layers=1", "model.wrench.query_heads=3",
                "data.val_datasets=[dexycb]", "loss.wrench.point_force_clamp_n=0.2",
                "training.max_epochs=3", "training.compile.enabled=false",
            ])
        model = SubstepForceModel(cfg).eval()
        with torch.no_grad():
            model.network.query_head.output.weight.normal_(std=0.03)
            model.network.query_head.output.bias.fill_(0.3)
        checkpoint = base / "model.ckpt"
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        # Both a stale local initialization path and an unset path must be
        # ignored when reconstructing a checkpoint (no pretraining download).
        cfg.model.wrench.pretrained = True
        cfg.model.wrench.pretrained_path = "/missing_initialization.safetensors"
        config = base / "config.yaml"
        OmegaConf.save(cfg, config, resolve=True)
        images = np.random.default_rng(41).integers(0, 256, (1, 2, 3, 32, 32), dtype=np.uint8)
        tensor = validate_images(images, cfg)
        with torch.inference_mode():
            expected = model(tensor)
            rebuilt = load_network(load_prediction_config(config), checkpoint)
            actual = rebuilt(tensor)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert cfg.model.wrench.pretrained is True
        bare = base / "network.pt"
        torch.save(model.network.state_dict(), bare)
        cfg.model.wrench.pretrained_path = None
        with torch.inference_mode():
            bare_network = load_network(cfg, bare)
            torch.testing.assert_close(bare_network(tensor)["force_30hz"], expected["force_30hz"], rtol=0, atol=0)
        broken = dict(model.network.state_dict())
        broken.pop(next(iter(broken)))
        torch.save(broken, base / "broken.pt")
        try:
            load_network(cfg, base / "broken.pt")
        except RuntimeError:
            pass
        else:
            raise AssertionError("partial checkpoint accepted")
        checks.append("strict Lightning/bare checkpoint parity, missing init ignored, partial state rejected")
        floats = ((tensor.float() / 255. - model.network.image_mean) / model.network.image_std).numpy()
        with torch.inference_mode():
            float_output = rebuilt(validate_images(floats, cfg, normalized=True))
        torch.testing.assert_close(float_output["force_30hz"], actual["force_30hz"], rtol=0, atol=0)

        axes = np.zeros((1, 2, 2, 778, 2, 3), np.float32)
        axes[..., 0, 0] = 1.
        axes[..., 1, 2] = 1.
        present = np.array([[[True, False], [True, True]]])
        geometry = {"point_force_scale": np.array(2., np.float32),
                    "vertex_local_yz_camera_10hz": axes, "hand_present_10hz": present}
        decoded = decode_predictions(actual, cfg, geometry)
        no_geometry = decode_predictions(actual, cfg)
        assert "vertex_force_camera_n_10hz" not in no_geometry
        batch = {
            "images": tensor, "point_force_scale": torch.tensor(2.),
            "hand_present_10hz": torch.from_numpy(present),
            "vertex_local_yz_camera_30hz": torch.from_numpy(axes).unsqueeze(2).expand(1, 2, 3, 2, 778, 2, 3),
            "vertex_force_camera_30hz": torch.zeros(1, 2, 3, 2, 778, 3),
            "vertex_lever_camera_30hz": torch.zeros(1, 2, 3, 2, 778, 3),
            "environment_wrench_camera_30hz": torch.zeros(1, 2, 3, 6),
            "hand_wrench_camera_30hz": torch.zeros(1, 2, 3, 6),
            "real_mask": torch.ones(1, 2, dtype=torch.bool),
            "real_mask_30hz": torch.ones(1, 2, 3, dtype=torch.bool),
            "force_valid_30hz": torch.ones(1, 2, 3, dtype=torch.bool),
            "contact_target_10hz": torch.zeros(1, 2, 2, 778),
            "contact_valid_10hz": torch.from_numpy(present),
            "proximity_target_10hz": torch.ones(1, 2, 2, 778),
            "proximity_valid_10hz": torch.from_numpy(present),
        }
        with torch.inference_mode():
            physical, _, _ = model._physical_outputs(expected["force_30hz"], batch, expected["contact_gate_10hz"])
        torch.testing.assert_close(decoded["vertex_force_camera_n_10hz"], physical[:, :, 0], rtol=0, atol=0)
        assert decoded["vertex_force_camera_n_10hz"][0, 0, 1].count_nonzero() == 0
        assert decoded["vertex_force_camera_n_10hz"].norm(dim=-1).max() <= 0.20001
        for invalid in ({}, dict(geometry, point_force_scale=np.array(0.)),
                        dict(geometry, vertex_local_yz_camera_10hz=axes * 2)):
            try:
                decode_predictions(actual, cfg, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid geometry accepted")
        try:
            validate_images(images.astype(np.float32), cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous float pixels accepted")
        checks.append("aligned physical force matches training consumer, no GT required, invalid scale/axes rejected")

        input_file, geometry_file = base / "input.npz", base / "geometry.npz"
        np.savez(input_file, images=images)
        np.savez(geometry_file, **geometry)
        output = base / "prediction.npz"
        with patch.object(sys, "argv", ["infer", "--config", str(config), "--ckpt", str(checkpoint),
                                       "--input", str(input_file), "--geometry", str(geometry_file), "--output", str(output)]):
            infer_main()
        assert output.with_suffix(".done").is_file()
        with np.load(output) as result:
            np.testing.assert_array_equal(result["contact_prob"], decoded["contact_prob"].numpy())
            np.testing.assert_array_equal(result["vertex_force_camera_n_10hz"], decoded["vertex_force_camera_n_10hz"].numpy())
        checks.append("tensor inference CLI writes NPZ/metadata/success marker")
        minimal_output = base / "minimal_prediction.npz"
        probe = '''
import importlib.abc
import sys
class NoTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pytorch_lightning', 'hydra', 'pyarrow'} or fullname == 'core.physics' or fullname.startswith('core.physics.'):
            raise RuntimeError('unexpected inference dependency: '+fullname)
sys.meta_path.insert(0, NoTrainingDependencies())
import torch
torch.set_num_threads(1)
from training.infer_tactohand import main
main()
'''
        with (base / "minimal.log").open("w") as stream:
            result = subprocess.run([sys.executable, "-c", probe,
                                     "--config", str(config), "--ckpt", str(checkpoint),
                                     "--input", str(input_file), "--output", str(minimal_output)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise AssertionError((base / "minimal.log").read_text()[-3000:])
        assert minimal_output.with_suffix(".done").is_file()
        checks.append("inference CLI runs with trainer/Hydra/data/physics imports prohibited")

        import cv2
        video = base / "synthetic.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10., (32, 16))
        assert writer.isOpened()
        for _ in range(3):
            image = np.zeros((16, 32, 3), np.uint8)
            image[..., 0] = 255
            writer.write(image)
        writer.release()
        frames, indices = read_video(video, frames=2, start_seconds=0., target_fps=10., image_size=32)
        np.testing.assert_array_equal(indices, [0, 1])
        assert frames.shape == (1, 2, 3, 32, 32)
        assert frames[0, 0, 2, 8:24].mean() == 255
        assert frames[0, 0, 0].max() == 0 and frames[0, 0, :, :8].max() == 0
        checks.append("CFR ordinal sampling, BGR-to-RGB and shared aspect-preserving letterbox")
        viewer_run = base / "runs" / "video_test"
        video_output = viewer_run / "predictions.npz"
        with patch.object(sys, "argv", ["infer", "--config", str(config), "--ckpt", str(checkpoint),
                                       "--video", str(video), "--frames", "2", "--start-seconds", "0.1",
                                       "--output", str(video_output)]):
            infer_main()
        with np.load(video_output) as result:
            np.testing.assert_array_equal(result["source_frame_indices"], [1, 2])
            np.testing.assert_allclose(result["target_times_s"], [0.1, 0.2])
        checkpoint_root = base / "checkpoints"
        checkpoint_root.mkdir()
        (checkpoint_root / "tactohand.ckpt").symlink_to(checkpoint)
        full_config = load_prediction_config(config)
        full_config.data.sequence_length = 2
        OmegaConf.save(full_config, checkpoint_root / "tactohand.yaml", resolve=True)
        with ViewerServer(("127.0.0.1", 0), viewer_run.parent, checkpoints=checkpoint_root,
                          device="cpu") as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(f"{url}/", timeout=5) as response:
                    html = response.read()
                assert b'id="run-select"' in html and b'id="payload"' not in html
                with urlopen(f"{url}/api/runs", timeout=5) as response:
                    assert json.load(response) == {"runs": [{"id": "video_test"}]}
                with urlopen(f"{url}/api/runs/video_test", timeout=5) as response:
                    payload = json.load(response)
                assert payload["times"] == [0.1, 0.2] and payload["indices"] == [1, 2]
                assert payload["metrics"]["force"]["unit"] == "normalized"
                request = Request(url + payload["video"]["url"], headers={"Range": "bytes=4-11"})
                with urlopen(request, timeout=5) as response:
                    assert response.status == 206 and response.read() == video.read_bytes()[4:12]
                assert not list(viewer_run.glob("*.html"))
                with urlopen(f"{url}/api/checkpoints", timeout=5) as response:
                    assert json.load(response) == {"checkpoints": [{"id": "tactohand.ckpt"}]}
                pending = viewer_run.parent / ".pending-progress_check"
                pending.mkdir()
                (pending / "inference.log").write_text("Window 0/4\nWindow 1/4\nWindow 2/4")
                assert server.jobs is not None
                server.jobs.update("progress_check", "running", "Loading", phase="inference")
                with urlopen(f"{url}/api/jobs/progress_check", timeout=5) as response:
                    assert json.load(response)["progress"] == {"completed": 1, "total": 4}
                query = urlencode({"checkpoint": "tactohand.ckpt", "filename": video.name})
                request = Request(f"{url}/api/jobs?{query}", data=video.read_bytes(),
                                  headers={"Content-Type": "application/octet-stream"})
                with urlopen(request, timeout=5) as response:
                    assert response.status == 202
                    new_run = json.load(response)["id"]
                deadline = time.monotonic() + 90
                while True:
                    with urlopen(f"{url}/api/jobs/{new_run}", timeout=5) as response:
                        job = json.load(response)
                    if job["status"] in {"complete", "failed"} or time.monotonic() > deadline:
                        break
                    time.sleep(.1)
                assert job["status"] == "complete", job
                saved = viewer_run.parent / new_run
                with np.load(saved / "predictions.npz") as result:
                    np.testing.assert_array_equal(result["source_frame_indices"], [0, 1, 2])
                    np.testing.assert_allclose(result["target_times_s"], [0., .1, .2])
                    assert result["contact_prob"].shape == (1, 3, 2, 778)
                assert (saved / "predictions.done").is_file()
                inference_log = (saved / "inference.log").read_text()
                assert "Window 0/2" in inference_log and "Window 2/2" in inference_log
            finally:
                server.shutdown()
                worker.join()
        checks.append("viewer streams video, accepts uploads and saves complete overlapping-window CPU predictions")
        missing_stats = SimpleNamespace(labels_root=None, processed_root=str(base), dataset_names=["missing"])
        with patch("core.tactile_labels.decode.load_label_stats", side_effect=FileNotFoundError):
            try:
                EvalHarness._skip_train_build(missing_stats, strict_stats=True)
            except FileNotFoundError as error:
                assert "normalization statistics" in str(error)
            else:
                raise AssertionError("missing original normalization stats accepted")
            EvalHarness._skip_train_build(missing_stats)
            assert missing_stats.train_ds.datasets == []
        checks.append("explicit-config normalization metadata required; legacy skip policy preserved")

        run = base / "historical_run"
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(cfg, run / ".hydra/config.yaml", resolve=True)
        (run / "checkpoints").mkdir()
        (run / "checkpoints/last.ckpt").write_bytes(checkpoint.read_bytes())
        assert resolve_checkpoint(run, "last") == run / "checkpoints/last.ckpt"
        with patch.object(EvalHarness, "_build_loader", return_value=[batch]), \
                patch("training.evaluation.harness._git_rev", return_value="not_recorded"):
            old = EvalHarness(run, checkpoint, ["dexycb"], ["contact", "proximity", "force"], device="cpu").run()
            standalone = EvalHarness(None, checkpoint, ["dexycb"], ["contact", "proximity", "force"],
                                     device="cpu", config_path=config).run()
            without_priors = OmegaConf.create(OmegaConf.to_container(cfg))
            without_priors.loss.haco.contact_statistics_path = "/absent/contact.npz"
            without_priors.loss.wrench.force_statistics_path = "/absent/force.npz"
            priorless_config = base / "priorless.yaml"
            OmegaConf.save(without_priors, priorless_config, resolve=True)
            priorless = EvalHarness(None, bare, ["dexycb"], ["contact"],
                                    device="cpu", config_path=priorless_config)._build_model()
            with torch.inference_mode():
                torch.testing.assert_close(priorless(tensor)["force_30hz"], expected["force_30hz"], rtol=0, atol=0)
            assert old["counts"] == standalone["counts"]
            for metric, value in old["per_dataset"]["dexycb"].items():
                np.testing.assert_allclose(value["value"], standalone["per_dataset"]["dexycb"][metric]["value"], equal_nan=True)
            eval_output = base / "eval"
            with patch.object(sys, "argv", ["eval", "--config", str(config), "--ckpt", str(checkpoint),
                                           "--device", "cpu", "--metrics", "contact", "--output", str(eval_output),
                                           "--processed-root", str(base / "relocated_data")]):
                eval_main()
            report = json.loads(next(eval_output.glob("*/tactohand/*.json")).read_text())
            assert report["run"] is None
            assert report["config"]["processed_root"] == str(base / "relocated_data")
        checks.append("legacy/explicit-config real metric parity on synthetic labels and CLI root override")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
    args.output.with_suffix(".done").touch()
    print(f"PASS: {len(checks)} prediction/evaluation contracts")


if __name__ == "__main__":
    main()
