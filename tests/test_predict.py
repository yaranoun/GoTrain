"""gotrain.predict (the MCP predict tool): CPU only, synthetic, no downloads, never a real cache or test split.

Bare (randomly initialized) networks are saved as checkpoints in the temp workspace; data.load_split / load_stats
return synthetic val arrays; the ledger, vault and registry files point at a temp dir for this module.
"""

import base64
import io
import json
import shutil

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import timm
import torch
from PIL import Image

import gotrain
from gotrain import ledger, predict, registry, state
from gotrain.engine import data, tasks
from gotrain.engine import segmentation as seg
from gotrain.engine.train import predict as engine_predict
from gotrain.engine.train import prepare, resolve_config
from gotrain.store import sha256_file

EU, CA = "m-eurosat", "m-cashew-plant"
CPU = torch.device("cpu")
_rng = np.random.default_rng(0)
STATS = {EU: (np.linspace(1000, 2500, 13).astype(np.float32), np.full(13, 600, np.float32)),
         CA: (np.linspace(800, 3000, 12).astype(np.float32), np.full(12, 700, np.float32))}
VAL = {
    EU: {"x": np.round(_rng.uniform(0, 4000, (4, 13, 32, 32))).astype(np.uint16), "y": np.arange(4) % 10,
         "n_classes": 10, "classes": tasks.CLASS_NAMES[EU], "channel_names": tasks.CHANNELS[EU]["names"]},
    CA: {"x": np.round(_rng.uniform(0, 4000, (3, 12, 64, 64))).astype(np.uint16),
         "y": _rng.integers(0, 7, (3, 64, 64)).astype(np.uint8), "n_classes": 7, "classes": tasks.CLASS_NAMES[CA],
         "channel_names": tasks.CHANNELS[CA]["names"]},
}
EU_CFG = {"backbone": "resnet18", "weights": "imagenet"}
CA_CFG = {"backbone": "resnet18", "weights": "imagenet", "decoder": "unet"}


def _fake_split(task, split):
    assert split == "val", f"predict asked for the {split} split"
    return {**VAL[task], "mean": STATS[task][0], "std": STATS[task][1]}


def _run(task, run_id, cfg, net=None, composite=None, stage="train"):
    """A finished run in the temp workspace with config.json + checkpoint.pt (or composite.json) and its ledger row."""
    d = gotrain.WORKSPACE / "runs" / task / run_id
    d.mkdir(parents=True, exist_ok=True)
    if composite is None:
        (d / "config.json").write_text(json.dumps({"task": task, **cfg}))
        torch.save(net.state_dict(), d / "checkpoint.pt")
        model_file = d / "checkpoint.pt"
    else:
        (d / "composite.json").write_text(json.dumps({"task": task, "mechanism": "ensemble", "members": composite}))
        model_file = d / "composite.json"
    rec = {"run_id": run_id, "task": task, "task_type": tasks.info(task)["type"], "stage": stage, "status": "done",
           "session_id": "pt-session", "config": cfg, "created_at": "2026-11-01T05:00:00Z",
           "artifacts": {"dir": str(d.relative_to(gotrain.WORKSPACE)), "checkpoint_sha256": sha256_file(model_file)}}
    ledger.append(ledger.LEDGER, rec)
    return d


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("predict")
    mp = pytest.MonkeyPatch()
    mp.setattr(ledger, "LEDGER", tmp / "ledger.jsonl")
    mp.setattr(ledger, "VAULT", tmp / "vault.jsonl")
    mp.setattr(registry, "REGISTRY", tmp / "registry.jsonl")
    mp.setattr(data, "load_split", _fake_split)
    mp.setattr(data, "load_stats", lambda task: STATS[task])
    torch.manual_seed(0)
    dirs = {}
    for name in ("pt-eu-a", "pt-eu-b"):
        dirs[name] = _run(EU, name, EU_CFG, timm.create_model("resnet18", pretrained=False, in_chans=13,
                                                               num_classes=10))
    dirs["pt-ca-a"] = _run(CA, "pt-ca-a", CA_CFG, smp.Unet("resnet18", encoder_weights=None, in_channels=12,
                                                             classes=7))
    members = [{"run_id": n, "run_dir": str(dirs[n].relative_to(gotrain.WORKSPACE)), "tta": n == "pt-eu-b",
                "checkpoint_sha256": sha256_file(dirs[n] / "checkpoint.pt")} for n in ("pt-eu-a", "pt-eu-b")]
    dirs["pt-eu-L2"] = _run(EU, "pt-eu-L2", {}, composite=members, stage="level2")
    yield dirs
    mp.undo()
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)


def _engine_probs(run_dir, task, x, tta):
    """Class probabilities of one run by the engine's own predict (engine.train.predict)."""
    raw = json.loads((run_dir / "config.json").read_text())
    raw.pop("task")
    cfg = resolve_config(raw, task)
    net = timm.create_model("resnet18", pretrained=False, in_chans=13, num_classes=10)
    net.load_state_dict(torch.load(run_dir / "checkpoint.pt", weights_only=True))
    mean, std = (torch.from_numpy(a) for a in STATS[task])
    return engine_predict(net, x[None].astype(np.float32), cfg, mean, std, CPU, tta=tta)[0]


def _probs(res):
    return np.array([res["probs"][c] for c in tasks.CLASS_NAMES[EU]])


def test_classification_probs_and_every_input_form(world, tmp_path):
    x = VAL[EU]["x"][0]
    ref = predict.predict("val:0", task=EU, model="pt-eu-a")
    assert ref["kind"] == "checkpoint" and ref["model_id"] == "pt-eu-a" and ref["tta"] is False
    assert set(ref["probs"]) == set(tasks.CLASS_NAMES[EU]) and abs(sum(ref["probs"].values()) - 1) < 1e-4
    assert list(ref["probs"].values()) == sorted(ref["probs"].values(), reverse=True)
    assert len(ref["top"]) == 5 and ref["top"][0] == list(next(iter(ref["probs"].items())))
    assert np.allclose(_probs(ref), _engine_probs(world["pt-eu-a"], EU, x, False), atol=1e-5)

    np.save(tmp_path / "chw.npy", x)
    np.save(tmp_path / "hwc.npy", x.transpose(1, 2, 0))
    buf = io.BytesIO()
    np.save(buf, x)
    b64 = "b64npy:" + base64.b64encode(buf.getvalue()).decode()
    for image in (str(tmp_path / "chw.npy"), tmp_path / "hwc.npy", b64, x.astype(np.float32)):
        res = predict.predict(image, model="pt-eu-a")  # the task comes from the model
        assert res["task"] == EU and np.allclose(_probs(res), _probs(ref), atol=1e-6)


def test_load_input_forms_and_refusals(world, tmp_path):
    assert np.array_equal(predict.load_input(EU, "val:2"), VAL[EU]["x"][2].astype(np.float32))
    # m-brick-kiln: all 13 on-disk channels -> its 10 kept bands, in cache order
    disk = np.stack([np.full((40, 40), i, np.float32) for i in range(13)])
    got = predict.load_input("m-brick-kiln", disk)
    assert got.shape == (10, 40, 40) and list(got[:, 0, 0]) == tasks.CHANNELS["m-brick-kiln"]["permutation"]
    # m-SA-crop-type is stored 8-bit: raw inputs get VALUE_SCALE (rounded), like the cache
    assert predict.load_input("m-SA-crop-type", np.full((12, 32, 32), 255, np.uint8)).max() == 10000
    # RGB tasks take PNGs
    rgb = _rng.integers(0, 256, (40, 48, 3)).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    assert np.array_equal(predict.load_input("m-pv4ger-seg", str(tmp_path / "rgb.png")), rgb.transpose(2, 0, 1))
    b64png = "b64png:" + base64.b64encode((tmp_path / "rgb.png").read_bytes()).decode()
    assert np.array_equal(predict.load_input("m-pv4ger-seg", b64png), rgb.transpose(2, 0, 1))
    with pytest.raises(ValueError, match="never an input"):
        predict.load_input(EU, "test:0")
    with pytest.raises(ValueError, match="out of range"):
        predict.load_input(EU, "val:99")
    with pytest.raises(ValueError, match="only RGB"):
        predict.load_input(EU, str(tmp_path / "rgb.png"))
    with pytest.raises(ValueError, match="takes 13 bands"):
        predict.load_input(EU, np.zeros((5, 32, 32)))
    with pytest.raises(ValueError, match="no such file"):
        predict.load_input(EU, str(tmp_path / "missing.npy"))


def test_composite_is_the_mean_of_its_members(world):
    x = VAL[EU]["x"][1]
    res = predict.predict("val:1", task=EU, model="pt-eu-L2")
    assert res["kind"] == "composite" and res["tta"] == [False, True]
    by_hand = (_engine_probs(world["pt-eu-a"], EU, x, False) + _engine_probs(world["pt-eu-b"], EU, x, True)) / 2
    assert np.allclose(_probs(res), by_hand, atol=1e-5)
    singles = [predict.predict("val:1", model=m, tta=t) for m, t in (("pt-eu-a", False), ("pt-eu-b", True))]
    assert np.allclose(_probs(res), (_probs(singles[0]) + _probs(singles[1])) / 2, atol=1e-5)
    # the tta argument does not override a composite's recorded member flags
    assert np.allclose(_probs(predict.predict("val:1", model="pt-eu-L2", tta=False)), _probs(res), atol=1e-6)


def test_segmentation_mask_fractions_and_overlay(world, tmp_path):
    out = tmp_path / "predictions"
    res = predict.predict("val:1", task=CA, model="pt-ca-a", tta=True, out_dir=out)
    assert res["kind"] == "checkpoint" and res["tta"] is True and res["shape"] == [64, 64]
    mask = np.load(res["mask_path"])
    assert [p.name for p in out.iterdir()] == [res["mask_path"].rsplit("/", 1)[1]]  # nothing else written
    assert mask.shape == (64, 64) and 0 not in np.unique(mask)
    assert "no data" not in res["class_fractions"] and abs(sum(res["class_fractions"].values()) - 1) < 1e-4
    # the engine's own TTA mask prediction
    net = smp.Unet("resnet18", encoder_weights=None, in_channels=12, classes=7)
    net.load_state_dict(torch.load(world["pt-ca-a"] / "checkpoint.pt", weights_only=True))
    cfg = resolve_config(dict(CA_CFG), CA)
    mean, std = (torch.from_numpy(a) for a in STATS[CA])
    ref = seg.predict(net, VAL[CA]["x"][1:2].astype(np.float32), lambda z: prepare(z, cfg, mean, std), CPU,
                      tta=True, ignore_index=0)[0]
    assert (ref == mask).mean() > 0.999
    png = Image.open(io.BytesIO(res["overlay_png"]))
    png.load()
    assert png.format == "PNG" and max(png.size) <= 1024
    # a non-square raw input with TTA keeps its shape
    wide = predict.predict(VAL[CA]["x"][0][:, :, :48].transpose(1, 2, 0), model="pt-ca-a", tta=True, out_dir=out)
    assert wide["shape"] == [64, 48] and np.load(wide["mask_path"]).shape == (64, 48)


def test_masks_go_to_the_job_workspace_by_default(world, tmp_path, monkeypatch):
    from gotrain import jobs

    monkeypatch.setattr(jobs, "JOB_WS", tmp_path / "mcp-ws")
    res = predict.predict("val:0", task=CA, model="pt-ca-a")
    assert res["mask_path"].startswith(str(tmp_path / "mcp-ws" / "predictions"))
    assert [p.name for p in (tmp_path / "mcp-ws").rglob("*") if p.is_file()] == [res["mask_path"].rsplit("/", 1)[1]]


def test_cuda_is_refused_while_the_gpu_lock_is_held(world, monkeypatch):
    def busy(owner, **info):
        raise state.Busy("the GPU is busy: held by a training chain")

    def no_gpu():
        raise AssertionError("predict touched CUDA without the GPU lock")

    monkeypatch.setattr(state, "gpu_lock", busy)
    monkeypatch.setattr(torch.cuda, "is_available", no_gpu)
    with pytest.raises(RuntimeError, match="GPU busy") as e:
        predict.predict("val:0", task=EU, model="pt-eu-a", device="cuda")
    assert not isinstance(e.value, state.Busy)


def test_resolve_model_production_then_vault_fallback(world, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY", tmp_path / "registry.jsonl")
    monkeypatch.setattr(ledger, "VAULT", tmp_path / "vault.jsonl")
    with pytest.raises(ValueError, match="no production model"):
        predict.resolve_model(EU)
    with pytest.raises(ValueError, match="give a task"):
        predict.resolve_model()
    with pytest.raises(ValueError, match="not m-cashew-plant"):
        predict.resolve_model(CA, "pt-eu-a")

    # vault fallback: the task's finalized run (no registry entry)
    ledger.append(ledger.VAULT, {"task": CA, "run_id": "pt-ca-a", "tta": False, "members": None,
                                 "checkpoint_sha256": sha256_file(world["pt-ca-a"] / "checkpoint.pt"),
                                 "test_metrics": {"miou": 0.5}, "entry_hash": "sha256:pt"})
    spec = predict.resolve_model(CA)
    assert (spec["run_id"], spec["model_id"], spec["kind"]) == ("pt-ca-a", "pt-ca-a", "checkpoint")
    assert spec["run_dir"] == world["pt-ca-a"] and spec["members"][0]["tta"] is False

    # registry production wins; "<task>:vN" resolves to the same model
    entry = registry.register("pt-eu-L2", "production")
    spec = predict.resolve_model(EU)
    assert (spec["model_id"], spec["run_id"], spec["kind"]) == (entry["model_id"], "pt-eu-L2", "composite")
    assert [(m["run_id"], m["tta"]) for m in spec["members"]] == [("pt-eu-a", False), ("pt-eu-b", True)]
    assert predict.resolve_model(model=entry["model_id"])["run_dir"] == world["pt-eu-L2"]
    res = predict.predict("val:3", task=EU)
    assert res["model_id"] == entry["model_id"]
    assert np.allclose(_probs(res), _probs(predict.predict("val:3", model="pt-eu-L2")), atol=1e-6)


def test_a_changed_checkpoint_is_refused(world, monkeypatch):
    rec = ledger.runs()["pt-eu-a"]
    monkeypatch.setattr(ledger, "LEDGER", ledger.LEDGER.with_name("ledger-tampered.jsonl"))
    ledger.append(ledger.LEDGER, {**rec, "artifacts": {**rec["artifacts"], "checkpoint_sha256": "sha256:0"}})
    with pytest.raises(RuntimeError, match="no longer matches"):
        predict.predict("val:0", model="pt-eu-a")
