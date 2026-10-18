# Data

Every dataset is a folder with the same layout, so the training code never changes:

```
data/<name>/train/images/*.png   data/<name>/train/masks/*.png    (same file stem)
data/<name>/val/...              data/<name>/test/...
```

A mask pixel is 0 (background) or anything > 0 (object); masks are stored as 0/255 PNGs
so they can be opened in any viewer. To use your own data, put it in this layout
(any image size: it is resized to `--size` when loaded) and pass `--data`.

## `unetseg-data shapes`: synthetic, offline, seconds

```bash
unetseg-data shapes --size 96 --n-train 800 --n-val 100 --n-test 100 --seed 0
```

Each image has 1 to 3 filled shapes (ellipse, rectangle, triangle, 5-7 sided polygon) on a
textured background. Built to need *spatial* reasoning, not a colour threshold:

* the object and the background are both textured (smooth noise + stripes, a
  checkerboard or pixel noise) with colours drawn from the **same** range;
* 0 to 2 thin unfilled outlines in the object's texture are drawn as **distractors**
  (not in the mask): a model that only matches texture will paint them;
* Gaussian noise and a light blur.

The object covers ~12% of the pixels, a small-object problem like the coursework's
salt masks, which makes pixel accuracy misleading (predicting nothing scores 88%).
Splits use different seeds, so no image appears in two splits. ~4 s for 1 000 images.

## `unetseg-data pets`: Oxford-IIIT Pet, real photos

```bash
unetseg-data pets --max-images 1200 --size 128 --seed 0
```

Oxford-IIIT Pet (Parkhi et al., 2012): 7 390 photos of 37 cat and dog breeds, each with
a trimap (1 = pet, 2 = background, 3 = border). The binary mask is
`pet = (trimap == 1) | (trimap == 3)`: the border band belongs to the animal's outline.

Only `--max-images` photos are kept, picked at random with `--seed` across all breeds
(the archive is sorted by breed, so "the first N" would be two or three breeds). The
19 MB annotation archive is downloaded to `data/raw/`; the 790 MB image archive is
**streamed** and only the selected JPEGs are decoded, so nothing big is written to disk
(about 3 minutes at 20 MB/s, 41 MB of PNGs for 1 200 photos at 128 px). Images are
resized to `--size`×`--size` (bilinear), masks with nearest neighbour; the split is
80 / 10 / 10. The pet covers ~46% of the pixels on average, the opposite regime from
the shapes. With `torchvision.datasets.OxfordIIITPet` the same data can be had, but it
downloads and unpacks the whole archive.

## Loading and augmentation (`unetseg.data`)

Each split is read once into memory as uint8 tensors (24 MB for the shapes, ~60 MB for
1 200 pets), so an epoch costs only the model. File names are **sorted** before anything
else, so splits and batches are the same on every machine.

Augmentation (training split only) must move the image and the mask together:

| back end | flag | ops |
|---|---|---|
| torchvision `transforms.v2` | `--augment torchvision` (default) | h-flip, v-flip, random 90° rotation, affine (±30°, ±6% shift, 0.9-1.1 scale) on half of the samples |
| albumentations | `--augment albumentations` (`pip install albumentations`) | the coursework pipeline: flips, RandomRotate90, affine ±45° |
| none | `--augment none` | |

With torchvision the mask is wrapped in `tv_tensors.Mask`, so geometric ops use
nearest-neighbour interpolation on it (a mask pixel stays 0 or 1) and bilinear on the
image. A test feeds an image that *is* its mask through the augmentation and checks that
they still agree. Images are normalised with the ImageNet mean/std, as in the coursework.
