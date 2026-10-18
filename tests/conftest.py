import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def shapes_dir(tmp_path_factory):
    from unetseg.datasets import write_shapes

    root = tmp_path_factory.mktemp("shapes")
    write_shapes(root, size=32, n_train=24, n_val=8, n_test=8, seed=1)
    return root
