"""Step 5: what each architectural switch changes, without training.

Builds every preset, prints its parameter count and forward time, then follows one
96x96 image through the U-Net and prints the shape at every stage (encoder halves the
resolution and doubles the channels, the decoder does the reverse, and with skip
connections each decoder stage receives twice the channels). Runs in seconds.

    python examples/05_switches_side_by_side.py
"""

import time

import torch

from unetseg.config import PRESETS, ModelConfig
from unetseg.models import build_model, count_parameters

torch.manual_seed(0)
torch.set_num_threads(1)
x = torch.randn(8, 3, 96, 96)
print(f"{'model':15s} {'skip':5s} {'down':8s} {'conv':7s} {'resid':6s} {'params':>9s} {'fwd (8 imgs)':>12s}")
for name in PRESETS:
    cfg = ModelConfig.from_preset(name, img_size=96)
    model = build_model(cfg).eval()
    with torch.no_grad():
        t = time.time()
        model(x)
        dt = time.time() - t
    sw = ("-", "-", "-", "-") if name == "fcn" else (
        str(cfg.skip), cfg.down, "double" if cfg.double else "single", str(cfg.residual))
    params = count_parameters(model) / 1e6
    print(f"{name:15s} {sw[0]:5s} {sw[1]:8s} {sw[2]:7s} {sw[3]:6s} {params:8.2f}M {dt * 1000:10.0f}ms")

print("\nshapes through `unet` (batch of 1):")
model = build_model(ModelConfig.from_preset("unet")).eval()
for name, module in [("stem", model.stem), *[(f"down{i + 1}", m) for i, m in enumerate(model.downs)],
                     *[(f"up{len(model.ups) - i}", m) for i, m in enumerate(model.ups)], ("head", model.head)]:
    def show(m, inp, out, n=name):
        skip = f" + skip {tuple(inp[1].shape[1:])}" if len(inp) > 1 else ""
        print(f"  {n:6s} in {tuple(inp[0].shape[1:])!s:16s}{skip} -> out {tuple(out.shape[1:])}")

    module.register_forward_hook(show)
with torch.no_grad():
    model(torch.randn(1, 3, 96, 96))
