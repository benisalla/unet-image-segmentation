import json

import numpy as np
import torch
from PIL import Image

from unetseg.config import ModelConfig, TrainConfig
from unetseg.losses import SegLoss
from unetseg.metrics import Confusion
from unetseg.models import build_model
from unetseg.predict import predict_image
from unetseg.train import build_parser, load_checkpoint, train


def test_unet_overfits_one_batch():
    """The basic sanity check: a small U-Net must memorise 8 shape masks."""
    from unetseg.data import normalize
    from unetseg.datasets import make_shape_sample

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pairs = [make_shape_sample(rng, 32) for _ in range(8)]
    x = torch.stack([normalize(torch.from_numpy(i.copy()).permute(2, 0, 1)) for i, _ in pairs])
    y = torch.stack([torch.from_numpy((m > 0).astype(np.float32))[None] for _, m in pairs])
    model = build_model(ModelConfig.from_preset("unet", base=8, depth=2))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = SegLoss("bce+dice")
    for _ in range(120):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    c = Confusion()
    with torch.no_grad():
        c.update(torch.sigmoid(model(x)) > 0.5, y > 0.5)
    assert c.summary()["iou"] > 0.8, c.summary()


def test_train_writes_a_run_folder_and_checkpoint_roundtrips(shapes_dir, tmp_path):
    mcfg = ModelConfig.from_preset("unet-residual", base=4, depth=2)
    tcfg = TrainConfig(data_dir=str(shapes_dir), out_dir=str(tmp_path / "run"), img_size=32, epochs=2, batch_size=8)
    summary = train(mcfg, tcfg, verbose=False)
    run = tmp_path / "run"
    for name in ["best.pt", "last.pt", "history.json", "curves.png", "summary.json", "metrics.json",
                 "thresholds.png", "confusion.png", "predictions.png"]:
        assert (run / name).exists(), name
    assert len(json.loads((run / "history.json").read_text())) == 2
    assert 0.0 <= summary["test"]["at_threshold"]["iou"] <= 1.0
    model, cfg, ckpt = load_checkpoint(run / "best.pt")
    assert cfg.residual and ckpt["train_config"]["img_size"] == 32
    prob = predict_image(model, Image.new("RGB", (40, 30), (120, 80, 60)), size=32)
    assert prob.shape == (30, 40) and 0 <= prob.min() <= prob.max() <= 1


def test_fcn_trains_through_the_same_loop(shapes_dir, tmp_path):
    mcfg = ModelConfig.from_preset("fcn", fcn_hidden=16)
    tcfg = TrainConfig(data_dir=str(shapes_dir), out_dir=str(tmp_path / "fcn"), img_size=32, epochs=1, batch_size=8)
    assert train(mcfg, tcfg, evaluate_test=False, verbose=False)["params"] > 0


def test_cli_switches_override_the_preset():
    a = build_parser().parse_args(["--model", "unet", "--no-skip", "--down", "strided", "--residual"])
    cfg = ModelConfig.from_preset(a.model, skip=a.skip, down=a.down, double=a.double, residual=a.residual)
    assert (cfg.skip, cfg.down, cfg.double, cfg.residual) == (False, "strided", True, True)
