"""`unetseg-data`: write the two built-in datasets to disk.

`shapes` (offline, seconds): random filled shapes (ellipses, rectangles, triangles,
polygons) on textured backgrounds. The object and the background are drawn from the same
colour range and both are textured, and thin unfilled outlines are added as distractors,
so a model has to use shape and texture, not one colour, to separate them.

`pets`: the Oxford-IIIT Pet dataset (Parkhi et al. 2012, 37 breeds, 7 349 photos) with
its trimap annotations turned into a binary mask: pet (trimap 1) and its border
(trimap 3) = 1, background (trimap 2) = 0. Only `--max-images` photos are kept:
the 790 MB image archive is streamed and only the selected files are decoded, so nothing
big is written to disk. Images and masks are resized to `--size` x `--size`.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PETS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets"


# ----------------------------------------------------------------------------- shapes
def _texture(rng: np.random.Generator, size: int, colour: np.ndarray) -> np.ndarray:
    """A colour field with low-frequency noise plus stripes or a checkerboard."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    coarse = rng.normal(size=(6, 6))
    noise = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC))
    kind = rng.integers(3)
    freq = rng.uniform(4, 14)
    angle = rng.uniform(0, math.pi)
    if kind == 0:
        pattern = np.sin(2 * math.pi * freq * (xx * math.cos(angle) + yy * math.sin(angle)))
    elif kind == 1:
        pattern = np.sign(np.sin(2 * math.pi * freq * xx) * np.sin(2 * math.pi * freq * yy))
    else:
        pattern = rng.normal(size=(size, size))
    field = 0.5 + 0.18 * noise + rng.uniform(0.05, 0.2) * pattern
    tint = colour[None, None, :] * field[..., None]
    return np.clip(tint, 0, 1)


def _random_shape(draw: ImageDraw.ImageDraw, rng: np.random.Generator, size: int, fill, width: int = 0):
    cx, cy = rng.uniform(0.15, 0.85, 2) * size
    r = rng.uniform(0.08, 0.25) * size
    kind = rng.integers(4)
    if kind == 0:
        rx, ry = r, r * rng.uniform(0.5, 1.0)
        box = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(box, fill=fill if not width else None, outline=fill if width else None, width=width)
    elif kind == 1:
        box = [cx - r, cy - r * rng.uniform(0.4, 1.0), cx + r, cy + r * rng.uniform(0.4, 1.0)]
        draw.rectangle(box, fill=fill if not width else None, outline=fill if width else None, width=width)
    else:
        n = 3 if kind == 2 else int(rng.integers(5, 8))
        start = rng.uniform(0, 2 * math.pi)
        pts = [
            (cx + r * rng.uniform(0.7, 1.0) * math.cos(start + 2 * math.pi * k / n),
             cy + r * rng.uniform(0.7, 1.0) * math.sin(start + 2 * math.pi * k / n))
            for k in range(n)
        ]
        if width:
            draw.line(pts + [pts[0]], fill=fill, width=width)
        else:
            draw.polygon(pts, fill=fill)


def make_shape_sample(rng: np.random.Generator, size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """One (H x W x 3 uint8 image, H x W uint8 {0,255} mask)."""
    bg = _texture(rng, size, rng.uniform(0.3, 1.0, 3))
    fg = _texture(rng, size, rng.uniform(0.3, 1.0, 3))
    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_img)
    for _ in range(int(rng.integers(1, 4))):
        _random_shape(draw, rng, size, 255)
    mask = np.asarray(mask_img) > 0
    img = np.where(mask[..., None], fg, bg)
    # distractors: thin outlines in the object colour, NOT part of the mask
    outline = Image.new("L", (size, size), 0)
    odraw = ImageDraw.Draw(outline)
    for _ in range(int(rng.integers(0, 3))):
        _random_shape(odraw, rng, size, 255, width=max(1, size // 48))
    o = (np.asarray(outline) > 0) & ~mask
    img = np.where(o[..., None], fg * 0.9, img)
    img = img + rng.normal(0, 0.03, img.shape)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(pil), (mask * 255).astype(np.uint8)


def write_shapes(out: Path, size: int = 96, n_train: int = 800, n_val: int = 100, n_test: int = 100, seed: int = 0):
    for split, n, offset in (("train", n_train, 0), ("val", n_val, 1), ("test", n_test, 2)):
        rng = np.random.default_rng([seed, offset])
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, mask = make_shape_sample(rng, size)
            Image.fromarray(img).save(out / split / "images" / f"{split}_{i:05d}.png")
            Image.fromarray(mask).save(out / split / "masks" / f"{split}_{i:05d}.png")
    print(f"wrote shapes to {out}: {n_train} train / {n_val} val / {n_test} test, {size}x{size}")


# ----------------------------------------------------------------------------- pets
def trimap_to_binary(trimap: np.ndarray) -> np.ndarray:
    """Oxford-IIIT Pet trimap: 1 = pet, 2 = background, 3 = border -> pet+border = 1."""
    return ((trimap == 1) | (trimap == 3)).astype(np.uint8)


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def write_pets(out: Path, max_images: int = 1000, size: int = 128, seed: int = 0, raw: Path | None = None,
               val_frac: float = 0.1, test_frac: float = 0.1):
    raw = raw or out.parent / "raw" / "oxford-iiit-pet"
    ann = _download(f"{PETS_URL}/annotations.tar.gz", raw / "annotations.tar.gz")
    def is_trimap(m) -> bool:
        name = Path(m.name)
        return m.isfile() and name.parent.name == "trimaps" and name.suffix == ".png" and not name.name.startswith("._")

    with tarfile.open(ann, "r:gz") as tar:  # pass 1: names only
        stems = sorted(Path(m.name).stem for m in tar if is_trimap(m))
    random.Random(seed).shuffle(stems)
    order = stems[:max_images]
    wanted = set(order)
    trimaps: dict[str, np.ndarray] = {}
    with tarfile.open(ann, "r:gz") as tar:  # pass 2: decode only the selected trimaps
        for m in tar:
            if is_trimap(m) and Path(m.name).stem in wanted:
                trimaps[Path(m.name).stem] = np.asarray(Image.open(io.BytesIO(tar.extractfile(m).read())))
    print(f"{len(stems)} annotated photos, keeping a seeded random subset of {len(wanted)}")

    n_val = int(round(val_frac * len(wanted)))
    n_test = int(round(test_frac * len(wanted)))
    split_of = {s: ("val" if i < n_val else "test" if i < n_val + n_test else "train") for i, s in enumerate(order)}
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    local_images = raw / "images.tar.gz"
    src = open(local_images, "rb") if local_images.exists() else urllib.request.urlopen(f"{PETS_URL}/images.tar.gz")
    print("streaming images.tar.gz (only the selected photos are decoded)")
    found = 0
    with src, tarfile.open(fileobj=src, mode="r|gz") as tar:
        for m in tar:
            stem = Path(m.name).stem
            if not m.isfile() or stem not in wanted or Path(m.name).name.startswith("._"):
                continue
            try:
                img = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
            except Exception:  # a few files in the archive are not valid JPEGs
                continue
            mask = Image.fromarray(trimap_to_binary(trimaps[stem]) * 255)
            split = split_of[stem]
            img.resize((size, size), Image.BILINEAR).save(out / split / "images" / f"{stem}.png")
            mask.resize((size, size), Image.NEAREST).save(out / split / "masks" / f"{stem}.png")
            found += 1
            if found == len(wanted):
                break
    counts = {s: len(list((out / s / "images").glob("*.png"))) for s in ("train", "val", "test")}
    print(f"wrote pets to {out}: {counts}, {size}x{size}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="unetseg-data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shapes", help="synthetic shapes on textured backgrounds (offline)")
    s.add_argument("--out", default="data/shapes")
    s.add_argument("--size", type=int, default=96)
    s.add_argument("--n-train", type=int, default=800)
    s.add_argument("--n-val", type=int, default=100)
    s.add_argument("--n-test", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("pets", help="Oxford-IIIT Pet subset with binary pet/background masks")
    q.add_argument("--out", default="data/pets")
    q.add_argument("--max-images", type=int, default=1000)
    q.add_argument("--size", type=int, default=128)
    q.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "shapes":
        write_shapes(Path(a.out), a.size, a.n_train, a.n_val, a.n_test, a.seed)
    else:
        write_pets(Path(a.out), a.max_images, a.size, a.seed)


if __name__ == "__main__":
    main()
