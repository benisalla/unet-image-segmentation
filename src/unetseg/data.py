"""Loading image/mask pairs and augmenting them together.

On-disk layout (written by `unetseg-data`, or bring your own):

    <root>/train/images/xxx.png   <root>/train/masks/xxx.png   (mask: 0 = background, >0 = object)
    <root>/val/...                <root>/test/...

Each split is read once into memory as uint8 tensors (a few tens of MB for the
datasets here), so an epoch costs only the model, not JPEG decoding.

Augmentation must move the image and the mask *together*: flipping the image but not
the mask teaches the model the wrong answer. Two back ends do that:

* `torchvision` (default, no extra install): `transforms.v2` with `tv_tensors.Mask`, so
  geometric ops use nearest-neighbour on the mask and bilinear on the image.
* `albumentations` (optional, `pip install albumentations`): the notebook's pipeline
  (flips, RandomRotate90, ShiftScaleRotate). Falls back to torchvision if missing.

Validation and test data are never augmented, only normalised.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_split(root: str | Path, split: str, size: int, limit: int | None = None):
    """Read `<root>/<split>` into (N x 3 x S x S uint8 images, N x 1 x S x S uint8 masks, names)."""
    img_dir = Path(root) / split / "images"
    mask_dir = Path(root) / split / "masks"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found: run `unetseg-data shapes` (or pets) first")
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)  # sorted: reproducible
    if limit:
        names = names[:limit]
    imgs, masks = [], []
    for name in names:
        img = Image.open(img_dir / name).convert("RGB")
        mask = Image.open(mask_dir / (Path(name).stem + ".png")).convert("L")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        if mask.size != (size, size):
            mask = mask.resize((size, size), Image.NEAREST)
        imgs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        masks.append(torch.from_numpy((np.asarray(mask) > 0).astype(np.uint8))[None])
    return torch.stack(imgs), torch.stack(masks), names


def normalize(img_uint8: torch.Tensor) -> torch.Tensor:
    x = img_uint8.float() / 255.0
    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std = torch.tensor(STD).view(-1, 1, 1)
    return (x - mean) / std


def denormalize(x: torch.Tensor) -> np.ndarray:
    """C x H x W normalised tensor -> H x W x 3 float array in [0, 1] (for plotting)."""
    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std = torch.tensor(STD).view(-1, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


class TorchvisionAugment:
    """Joint flips, random 90-degree rotation and a small affine, with torchvision v2."""

    def __init__(self):
        from torchvision import tv_tensors
        from torchvision.transforms import v2

        self.tv_tensors = tv_tensors
        self.tf = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomApply([v2.RandomAffine(degrees=30, translate=(0.06, 0.06), scale=(0.9, 1.1))], p=0.5),
            ]
        )

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = int(torch.randint(0, 4, ()))  # RandomRotate90
        img, mask = torch.rot90(img, k, (1, 2)), torch.rot90(mask, k, (1, 2))
        img, mask = self.tf(self.tv_tensors.Image(img), self.tv_tensors.Mask(mask))
        return img.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)


class AlbumentationsAugment:
    """The notebook's albumentations pipeline (minus Resize/Normalize, done elsewhere)."""

    def __init__(self):
        import albumentations as A

        self.tf = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
            ]
        )

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.tf(image=img.permute(1, 2, 0).numpy(), mask=mask[0].numpy())
        return (
            torch.from_numpy(np.ascontiguousarray(out["image"])).permute(2, 0, 1),
            torch.from_numpy(np.ascontiguousarray(out["mask"]))[None],
        )


def make_augment(kind: str):
    if kind == "none":
        return None
    if kind == "albumentations":
        try:
            return AlbumentationsAugment()
        except ImportError:
            warnings.warn("albumentations is not installed, using the torchvision augmentations", stacklevel=2)
    elif kind != "torchvision":
        raise ValueError(f"augment must be none, torchvision or albumentations, got {kind!r}")
    return TorchvisionAugment()


class SegDataset(Dataset):
    """In-memory image/mask pairs -> (normalised float image, float {0,1} mask)."""

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, augment=None, names=None):
        self.images, self.masks, self.augment = images, masks, augment
        self.names = names or [str(i) for i in range(len(images))]

    @classmethod
    def from_folder(cls, root, split, size, augment=None, limit=None):
        imgs, masks, names = load_split(root, split, size, limit)
        return cls(imgs, masks, augment, names)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img, mask = self.images[i], self.masks[i]
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return normalize(img), mask.float()

    def foreground_fraction(self) -> float:
        return float(self.masks.float().mean())


def make_loaders(root, size, batch_size, augment="torchvision", max_train=None, seed=0):
    """Train (augmented, shuffled), val and test loaders. Each split gets its own dataset
    object, so turning augmentation off for val/test cannot touch the training set."""
    train = SegDataset.from_folder(root, "train", size, make_augment(augment), max_train)
    val = SegDataset.from_folder(root, "val", size)
    test_dir = Path(root) / "test"
    test = SegDataset.from_folder(root, "test", size) if test_dir.is_dir() else val
    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen, drop_last=len(train) > batch_size),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
