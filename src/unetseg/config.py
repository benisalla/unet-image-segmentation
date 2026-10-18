"""Configuration dataclasses and the model presets.

The coursework notebook had three boolean switches on one `UNet` class
(`is_resnet`, `is_max_pool`, `is_double`) plus a separate fully connected model.
Here each switch has a plain name, and every combination the notebook trained is a
named preset that `--model` selects:

| preset          | skip | down     | conv   | residual | notebook switches                         |
|-----------------|------|----------|--------|----------|-------------------------------------------|
| `unet`          | yes  | max-pool | double | no       | is_resnet=True,  is_max_pool=True,  is_double=True  |
| `unet-strided`  | yes  | strided  | double | no       | is_resnet=True,  is_max_pool=False, is_double=True  |
| `unet-single`   | yes  | max-pool | single | no       | is_resnet=True,  is_max_pool=True,  is_double=False |
| `unet-residual` | yes  | max-pool | double | yes      | (new: real residual blocks)               |
| `segnet`        | no   | max-pool | double | no       | is_resnet=False, is_max_pool=True,  is_double=True  |
| `fcn`           | -    | -        | -      | -        | the notebook's fully connected `FCN`      |

Note: the notebook's `is_resnet` flag switched the U-Net *skip connections*
(concatenation), not residual blocks. `--residual` here adds real residual blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PRESETS: dict[str, dict] = {
    "unet": dict(skip=True, down="maxpool", double=True, residual=False),
    "unet-strided": dict(skip=True, down="strided", double=True, residual=False),
    "unet-single": dict(skip=True, down="maxpool", double=False, residual=False),
    "unet-residual": dict(skip=True, down="maxpool", double=True, residual=True),
    "segnet": dict(skip=False, down="maxpool", double=True, residual=False),
    "fcn": dict(),
}


@dataclass
class ModelConfig:
    """Architecture. `name` is a preset; the other fields override it."""

    name: str = "unet"
    in_ch: int = 3
    base: int = 16  # channels of the first block (the notebook used 64)
    depth: int = 4  # number of down-sampling steps (the notebook used 4)
    skip: bool = True
    down: str = "maxpool"  # "maxpool" | "strided"
    double: bool = True
    residual: bool = False
    img_size: int = 96  # only used by `fcn` (its layers depend on the pixel count)
    fcn_hidden: int = 256  # width of the fcn bottleneck stack

    @classmethod
    def from_preset(cls, name: str, **overrides) -> ModelConfig:
        if name not in PRESETS:
            raise ValueError(f"unknown model {name!r}, choose from {sorted(PRESETS)}")
        values = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(name=name, **values)


@dataclass
class TrainConfig:
    """Optimisation and data loading."""

    data_dir: str = "data/shapes"
    out_dir: str = "runs/unet"
    img_size: int = 96
    epochs: int = 8
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    loss: str = "bce+dice"  # "bce" | "dice" | "bce+dice"
    threshold: float = 0.5
    augment: str = "torchvision"  # "none" | "torchvision" | "albumentations"
    max_norm: float = 5.0
    max_train: int | None = None  # cap on train images (quick runs)
    threads: int = 1
    seed: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
