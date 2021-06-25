"""
pytest test_x_dataio.py
missing tests strided mgrid, Datasets, Dataloader
"""
import numpy as np
import torch
from x_dataio import expand_igrid, flatten_igrid, mgrid_from_igrid, get_mgrid

# pylint: disable=no-member
def rnd_sidelens(sizes=(5,50), dims=(2,3,4)):
    out = []
    for dim in dims:
        out += [torch.randint(sizes[0], sizes[1], (dim,))]
    return out

def test_mgrids(sidelen_type="tensor"):

    device = "cpu" if sidelen_type != "cuda2" else "cuda"
    for sidelen in rnd_sidelens():
        prod = sidelen.prod().item()
        if sidelen_type in "list":
            sidelen = sidelen.tolist()
        elif sidelen_type in "tuple":
            sidelen = tuple(sidelen.tolist())
        elif sidelen_type in "ndarray":
            sidelen = sidelen.numpy()
        elif sidelen_type in "cuda":
            sidelen = sidelen.cuda()

        m = get_mgrid(sidelen, device=device)

        assert len(m) == prod, f"sidelen {sidelen}"
        assert m.shape[1] == len(sidelen), f"sidelen {sidelen}"

        i = get_mgrid(sidelen, indices=True, device=device)
        f = get_mgrid(sidelen, indices=True, flat=True, device=device)

        assert i.shape == m.shape
        assert len(i) == len(f), f"sidelen {sidelen}"

        f2 = flatten_igrid(i, sidelen=sidelen)
        assert torch.allclose(f2, f), f"sidelen {sidelen}"

        i2 = expand_igrid(f, sidelen)
        assert torch.allclose(i2, i), f"sidelen {sidelen}"

        m2 = mgrid_from_igrid(i2, sidelen)
        assert torch.allclose(m2, m, atol=1e-6), f"sidelen {sidelen}"

def test_mgrids_list():
    test_mgrids("list")

def test_mgrids_tuple():
    test_mgrids("tuple")

def test_mgrids_array():
    test_mgrids("ndarray")

def test_mgrids_cuda():
    test_mgrids("cuda")

def test_mgrids_cuda2():
    test_mgrids("cuda2")