"""
mgrid functions - mgrid is siren's indexing scheme returning a grid of shape
    (positions, indexing_dim)
    it differs from numpy or torch meshgrid in the index ordering

subset of modified functions

    >>> get_mgrid(sidelen, ranges) # is generalized n dimension grid

    e.g get_mgrid([10,20]) - will generate a 2d grid with 200 (x,y) positions
        get_mgrid([10,20,30,40]) - will generate a 4d grid with (x,y,z,w) positions

    arg ranges (list(list,)) allows to return a contiguous subgrid on sub range, ie not -1,1 but arbitrary
    e.g.
        get_mgrid([10,20,30], ranges=[[3,6],[2,5],[1,3]]) returns a 3,3 subgrid within the 10*20*30 grid

    arg indices (bool) returns mgrid as permutation indices, or permutations in range


    since RAM will quickly fill out, ranges weere added to mgrid
    e.g. get_mgrid([10,20,30], [[5,10],[20,25],[30,35]])
            only generates a 25 position (x,y,z) subgrid within the 6000 positions

    >>> get_subgrid(indices, sidelen) returns positions only of indices within subgrid

    e.g. get_subgrid([34,1231,2], [10,20,30]) returns a (3,3) array

"""
import os
import os.path as osp

import numpy as np
from PIL import Image
# import skvideo.io
import vidi
import torch
from torch.utils.data import Dataset
from x_log import logger


# pylint: disable=no-member
# pylint: disable=not-callable

def get_mgrid(sidelen, ranges=None, indices=False, strides=1, flat=False, device="cpu"):
    """  nd generic dim meshgrid
    mesh grid indexing [-1,...,1] per dimension

    Breaks get_mgrid sortcut, needs to specifically pass each dimension,
    ie. (sidelen=[256,256]) for a square image, insetad of (sidelen=256, dim=2)
    Args
        sidelen     list, grid steps per dimension, tuples, ints or other iterables will fail
        ranges      list of ranges [None] | [[from:to(exlusive)], ....]
        indices     returns permutation indices [0,n] instead of [-1,1]
        strides     list, stride between samples

    Examples:
        >>> get_mgrid(sidelen=[121,34,34]) # return full meshgrid
        >>> get_mgrid(sidelen=[121,34,34], indices=True) # mesh grid indices

        >>> get_mgrid(sidelen=[121,34,34], ranges=[[0,10]]) # slice in dim 0
        >>> get_mgrid(sidelen=[121,34,34], ranges=[[0,10],[3,4],[5,6]]) # slice all dims
        >>> get_mgrid(sidelen=[121,34,34], ranges=[None, None,[5,6]]) # slice last dim

        >>> get_mgrid(sidelen=[121,34,34], strides=[6, 3, 3]) # strided grid
        >>> get_mgrid(sidelen=[121,34,34], strides=[6, 3, 3], indices) # strided grid

    """
    # sidelen = np.asarray(sidelen)
    sidelen = torch.as_tensor(sidelen).cpu()
    strides = _asarray(strides, len(sidelen))

    # if ranges is None:
    #     return _get_mgrid(sidelen, indices=indices, strides=strides)

    # sidelen in range
    ranges, sublen,  = _check_ranges(sidelen, ranges)
    # strided sidelen
    # sublen = np.ceil(sublen/strides).astype(int)
    sublen = torch.ceil(sublen/strides).to(dtype=torch.int64)

    out = []
    _offset = 0 if indices else 1
    for i, side in enumerate(sidelen):
        # grid step #
        step = 1 if indices else 2/(side - 1)
        # repeats
        pre = sublen[:i].prod() # prod(pre+post) = prod(sidelen)
        post = sublen[i+1:].prod()

        out += [((torch.arange(ranges[i][0], ranges[i][1], strides[i])*step) - _offset
                ).repeat_interleave(post).repeat(pre).view(-1,1)]
    out = torch.cat(out, dim=1)
    if indices:
        if flat:
            out = flatten_igrid(out, sidelen)
            # out  = out.mul(torch.tensor([sidelen[i:].prod() for i in range(1, len(sidelen)+1)])).sum(dim=1)
        out = out.to(dtype=torch.int64)
    return out.to(device=device)

def expand_igrid(indices, sidelen):
    """ expand index grid
    """
    sidelen = torch.as_tensor(sidelen).to(device=indices.device)
    return torch.stack([indices//sidelen[i+1:].prod()%sidelen[i]
                        for i in range(len(sidelen))], dim=1)

def flatten_igrid(grid, sidelen):
    """ flatten index grid
    """
    sidelen = torch.as_tensor(sidelen)
    return grid.mul(torch.tensor([sidelen[i:].prod() for i in
                                  range(1, len(sidelen)+1)], device=grid.device)).sum(dim=1)

def mgrid_from_igrid(indices, sidelen):
    """
    """
    sidelen = torch.as_tensor(sidelen).to(device=indices.device)
    return 2*indices/(sidelen-1) - 1

def _aslist(iterable):
    if isinstance(iterable, (list, tuple)):
        return list(iterable)
    if isinstance(iterable, np.ndarray, torch.Tensor):
        return iterable.tolist()
    assert isinstance(iterable, list), f"convert to list, {type(iterable)} not supported"

def _asarray(item, size):
    if isinstance(item, (int, np.int64)):
        item = torch.tensor([item for _ in range(size)])
    else:
        item = torch.as_tensor(item)
    return item


    # if isinstance(item, np.ndarray):
    #     pass
    # elif isinstance(item, (list, tuple)):
    #     item = np.asarray(item)
    # elif isinstance(item, (int, np.int)):
    #     item = np.array([item for _ in range(size)])
    # elif isinstance(item, torch.Tensor):
    #     item = item.cpu().numpy()
    # else:
    #     raise NotImplementedError(f"wrong type {type(item)}")
    # if len(item) == 1:
    #     item = np.concatenate([item for _ in range(size)])
    # return item

def _check_ranges(sidelen, ranges):
    """ sanity check, range between 0,sidelen[i]
    """
    # sidelen  = _aslist(sidelen)
    # sidelen = np.asarray(sidelen)
    sidelen = torch.as_tensor(sidelen)

    sublen = []
    ranges = _aslist(ranges) if isinstance(ranges, (np.ndarray, list, tuple, torch.Tensor)) else []

    for i in range(len(ranges), len(sidelen)):
        ranges += [[0, sidelen[i]]]

    for i, rng in enumerate(ranges):
        if not isinstance(rng, list) or not len(rng) == 2:
            ranges[i] = [0, sidelen[i]]
        else:
            ranges[i][0] = max(0, ranges[i][0])
            ranges[i][1] = min(ranges[i][1], sidelen[i])
            ranges[i][1] = max(ranges[i][1], ranges[i][0]+1)
        sublen += [ranges[i][1] - ranges[i][0]]
    # ranges = np.asarray(ranges)
    # sublen = np.asarray(sublen)
    ranges = torch.as_tensor(ranges).cpu()
    sublen = torch.as_tensor(sublen).cpu()
    return ranges, sublen

def get_subsidelen(sidelen, max_samples):
    """ returns a dictionary of step: [sidelen,..] <= max samples
    """
    sidelen = np.asarray(sidelen)
    pyr = get_stride_tree(sidelen, max_samples)

    out = {pyr[0]: torch.as_tensor([sidelen//pyr[-1]])}
    for i in range(1, len(pyr)):
        out[pyr[i]] = grid_permutations(sidelen//pyr[len(pyr)-1-i], pyr[i])
    return out

def get_stride_tree(sidelen, max_samples):
    """  calculates sparse occupancy matrices strides
    current: uniformely spaced, but SHOULD bias towards importance of dimension connectivity

    max stride  loads the entire dataset, sparsest
    min stride  loads contiguous block, densent, smallest.

    """
    sidelen = torch.as_tensor(sidelen)
    elems = sidelen.prod()
    offset = elems/max_samples
    if offset <= 1:
        return torch.tensor([1])
    top = torch.ceil(offset**(1/len(sidelen)))
    strides = 2**torch.arange(torch.floor(torch.log2(top)), -1,-1).to(dtype=torch.int64)
    #.to(dtype=torch.int64)
    if top > strides[0]:
        strides = torch.sort(torch.cat([top.view(1).to(dtype=torch.int64), strides]), descending=True)[0]
    return strides

def get_sparse_grid(sidelen, strides):
    if isinstance(strides, int):
        strides = [strides for _ in range(len(sidelen))]

    indices = get_mgrid(sidelen, strides=strides, indices=True)
    offsets = get_mgrid(strides, indices=True)
    return indices, offsets

def get_inflatable_grid(sidelen, strides):
    if isinstance(strides, int):
        strides = [strides for _ in range(len(sidelen))]
    strides = torch.as_tensor(strides)
    sidelen = torch.as_tensor(sidelen)

    indices = get_mgrid(sidelen/strides, indices=True)
    offsets = get_mgrid(strides, indices=True)
    return indices, offsets

    # handled by datalpader
    # shuffled_perms = torch.randperm(np.prod(strides))

def grid_permutations(sidelen, sizes):
    """  full set of permutations on  data size sidelen, of sizes
    Eg
    sidelen = [40,60]
    sizes = [2,2]
    ranges = [[[0,20],[0,30]], [[0,20][30,60]], [[20,40]],[0,30]], [[20,40]][30,60]]
    """
    if isinstance(sizes, (int, np.int64)):
        sizes = [sizes for i in range(len(sidelen))]
    ranges = [[],[],[]]
    for i, side in enumerate(sidelen):
        for j in range(sizes[i]):
            ranges[i].append([side - (sizes[i]-j)*side//sizes[i], side - (sizes[i]-j-1)*side//sizes[i]])

    _perm = get_mgrid(sizes, indices=True)
    for i in range(1, len(_perm[0])): # flattern
        _perm[:,i] += 1 +_perm[:,i-1].max()

    return torch.as_tensor(ranges).view(-1,2)[_perm]


# ###
# # Datasets
#
def sparse_image_list(folder, frame_range=None, extensions=(".png", ".jpg", ".jpeg")):
    """
    Args
    """
    images = sorted([f.path for f in os.scandir(folder)
                     if f.name[-4:].lower() in extensions])
    if frame_range is None:
        return images

    subset = []
    if isinstance(frame_range, int):
        frame_range = range(frame_range)
    elif (isinstance(frame_range, (list, tuple)) and len(frame_range) < 4):
        frame_range = range(*frame_range)

    for i in frame_range:
        if i < len(images):
            subset.append(images[i])

    return subset


class VideoDataset(Dataset):
    """ VideoDataset with variations on loading grid.

    Example:
        config = 'x_periment_scripts/eclipse_512_sub.yml'
        opt = x_utils.read_config(config)
        opt.sample_size = x_utils.estimate_samples(x_utils.GPUse().available, grad=1, **opt.siren)*2
        opt.shuffle=False
        kw = {k:opt[k] for k  in opt if k in list(inspect.signature(VideoDataset.__init__).parameters)[1:]}
    """
    def __init__(self, data_path, frame_range=None,
                 sample_fraction=1., sample_size=None,
                 strategy=-1, loglevel=20, device="cpu"):
        """
        Args
            sample_fraction     float(1.)   sample_size = data size * sample_fraction
            sample_size         int [None]  overrides sample fraction number of samples load

            # loading strategy in original dataset works best: single data array per epoch.
            strategy     int [-1]    -1: fully random samples, single iter per epoch
                                    0: fully random samples, all samples for the data, per per epoch
                                    1: shuffled random samples, all a samples for tehd ata
                                    2: complete sparsest sets, and dense set blocks, all samples 2x per epoch
                                    # 2 is no good, random wwrks better, 0,1,-1
        """
        super().__init__()
        log = logger("VideoDataset", level=loglevel)
        self.data = None
        self.sidelen = None
        self.mgrid = None

        self.sample_size = sample_size # naming change N_samples, overrides sample_fraction
        self.sample_fraction = sample_fraction
        self.sampler = None         # defined in strategy: 1
        self.grid_indices = None    # defined in strategy: 2
        self.grid_offset = None     # defined in strategy: 2
        self.strides = None         # defined in strategy: 2
        self.strategy = strategy

        as_centered_tensor = lambda x, device="cpu": torch.as_tensor((np.asarray(x, dtype=np.float32) - 127.5)/127.5, device=device)

        if 'npy' in data_path:
            self.data = torch.as_tensor(np.load(data_path), device=device)
        elif 'mp4' in data_path:
            # self.data = as_centered_tensor(skvideo.io.vread(data_path), device=device)
            self.data = as_centered_tensor(vidi.ffread(data_path), device=device)

        elif osp.isdir(data_path):
            images = sparse_image_list(data_path, frame_range)
            self.data = torch.stack([as_centered_tensor(Image.open(image), device=device) for image in images], dim=0)

        else:
            raise NotImplementedError("mp4, mov not yet re implemented")

        self.sidelen = torch.as_tensor(self.data.shape[:-1])
        self.channels = self.data.shape[-1]
        self.data = self.data.view(-1, self.data.shape[-1])
        log[0].info(f"Loaded data, sidelen: {self.sidelen.tolist()}, channels {self.channels}")
        log[0].info(f"         => reshaped to: {tuple(self.data.shape)}")

        if sample_size is not None:
            self.sample_fraction = min(1, (sample_size / self.sidelen.prod()).item())
        else:
            self.sample_size = max(1, int(self.sample_fraction * self.sidelen.prod()))

        log[0].info(" max sample_size, {}, fraction, {}".format(self.sample_size,
                    round(self.sample_fraction, 4)))

        # load entire data per epoch if it fits
        if self.sample_fraction >= 1:
            self.sample_fraction = 1
            self.mgrid = get_mgrid(self.sidelen)
            self.strategy = 0
            log[0].info(f" strategy: {self.strategy}, load grid {tuple(self.mgrid.shape)} in one step")

        # strategies 2+ sample ordered grids
        if self.strategy == 2:
            stride_pyramid = get_stride_tree(self.sidelen, self.sample_size)
            self.strides = torch.tensor([stride_pyramid[0] for _ in range(len(self.sidelen))])
            self.grid_indices, self.grid_offset =  get_inflatable_grid(self.sidelen, self.strides)
            self.sample_size = self.grid_indices.shape[0]
            log[0].info(f" strategy: {self.strategy}, iters: {len(self.grid_offset) * 2}")
            log[0].info(f"    strides: {self.strides.tolist()}, max mgrid block: {self.grid_indices[-1].tolist()}")
            log[0].info(f"    -> sample size: {self.sample_size}")


        elif self.strategy == 1:
            self.sampler = torch.randperm(self.sidelen.prod())
            log[0].info(f" strategy: {self.strategy}, randperm iters: {int(1/self.sample_fraction)}")

        # original, 1 iter per epoch
        elif self.strategy == -1:
            log[0].info(f" strategy: {self.strategy}, single sample per epoch")


    def shuffle(self):
        """ strategy 1 needs to shuffle indices"""
        if self.sampler is not None:
            self.sampler = torch.randperm(self.sidelen.prod())

    def __len__(self):
        if self.strategy == -1:
            return 1 # original strategy outputs single sample per
        elif self.strategy == 2:
            return len(self.grid_offset) * 2
        return int(1/self.sample_fraction)

    def _item(self, idx):
        with torch.no_grad():
            if self.sample_fraction < 1.:
                # non repeating sample grid/ sparsest and densest blocks
                if self.strategy == 2:
                    if idx < len(self.grid_offset): # sparsest
                        coords = self.grid_indices * self.strides + self.grid_offset[idx]
                    else: # inflate contiguous minigrids to offset idx
                        coords = self.grid_offset.max(dim=0).values * self.grid_offset[idx%len(self.grid_offset)] + self.grid_indices
                    data = self.data[flatten_igrid(coords, self.sidelen)].view(-1, self.channels)
                    coords = 2*coords/(self.sidelen-1) - 1
                else:
                    # possibly repeating
                    if self.strategy < 1:
                        coords = torch.randint(0, self.data.shape[0], (self.sample_size,))
                    # non repeating shuffled samples
                    elif self.strategy == 1:
                        coords = self.sampler[idx: idx+self.sample_size]

                    data = self.data[coords, :]
                    coords = 2*expand_igrid(coords, self.sidelen)/(self.sidelen-1) - 1
            else:
                coords = self.mgrid
                data = self.data
        return coords, data

    def __getitem__(self, idx):
        """
        """
        coords, data = self._item(idx)
        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class VideoDataset2(VideoDataset):
    """ VideoDataset returning (input,target) pairs
    """
    def __init__(self, data_path, frame_range=None,
                 sample_fraction=1., sample_size=None,
                 strategy=-1, loglevel=20, device="cpu"):
        super().__init__(data_path, frame_range, sample_fraction,
                         sample_size, strategy, loglevel, device)

    def __getitem__(self, idx):
        """
        """
        return self._item(idx)


