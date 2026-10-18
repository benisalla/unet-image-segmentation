import numpy as np
import torch

from unetseg.data import SegDataset, TorchvisionAugment, load_split, make_augment, make_loaders, normalize
from unetseg.datasets import make_shape_sample, trimap_to_binary


def test_shapes_are_deterministic_binary_and_not_trivial():
    a_img, a_mask = make_shape_sample(np.random.default_rng(3), 64)
    b_img, b_mask = make_shape_sample(np.random.default_rng(3), 64)
    assert np.array_equal(a_img, b_img) and np.array_equal(a_mask, b_mask)
    assert a_img.shape == (64, 64, 3) and a_img.dtype == np.uint8
    assert set(np.unique(a_mask)) <= {0, 255}
    fracs = [(make_shape_sample(np.random.default_rng(s), 64)[1] > 0).mean() for s in range(20)]
    assert 0.02 < np.mean(fracs) < 0.5


def test_trimap_border_counts_as_pet():
    tri = np.array([[1, 2], [3, 2]], dtype=np.uint8)
    assert trimap_to_binary(tri).tolist() == [[1, 0], [1, 0]]


def test_augmentation_moves_image_and_mask_together():
    torch.manual_seed(0)
    mask = torch.zeros(1, 32, 32, dtype=torch.uint8)
    mask[:, 4:14, 6:20] = 1
    img = (mask * 255).repeat(3, 1, 1)  # the image *is* the mask
    aug = TorchvisionAugment()
    for _ in range(20):
        i2, m2 = aug(img, mask)
        assert i2.shape == (3, 32, 32) and m2.shape == (1, 32, 32)
        assert set(m2.unique().tolist()) <= {0, 1}  # nearest-neighbour on masks
        agree = ((i2[0] > 127) == (m2[0] > 0)).float().mean()
        assert agree > 0.97


def test_albumentations_falls_back_or_works():
    aug = make_augment("albumentations")
    img = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
    mask = torch.zeros(1, 16, 16, dtype=torch.uint8)
    i2, m2 = aug(img, mask)
    assert i2.shape == (3, 16, 16) and m2.shape == (1, 16, 16)
    assert make_augment("none") is None


def test_folder_loading_and_splits_are_separate(shapes_dir):
    imgs, masks, names = load_split(shapes_dir, "train", 32)
    assert imgs.shape == (24, 3, 32, 32) and masks.shape == (24, 1, 32, 32)
    assert names == sorted(names)
    tr, va, te = make_loaders(shapes_dir, 32, 8)
    assert tr.dataset.augment is not None
    assert va.dataset.augment is None and te.dataset.augment is None  # notebook bug: val/test were augmented
    x, y = va.dataset[0]
    assert torch.allclose(x, normalize(va.dataset.images[0]))
    assert y.dtype == torch.float32 and set(y.unique().tolist()) <= {0.0, 1.0}


def test_dataset_without_augmentation_is_stable(shapes_dir):
    ds = SegDataset.from_folder(shapes_dir, "val", 32)
    assert torch.equal(ds[0][0], ds[0][0])
