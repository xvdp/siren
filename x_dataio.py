"""
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
from numpy.random import sample
import torch
from torch.utils.data import Dataset


# pylint: disable=no-member
# pylint: disable=not-callable


def get_mgrid(sidelen, ranges=None):
    """  nd generic dim meshgrid
    mesh grid indexing [-1,...,1] per dimension

    Breaks get_mgrid sortcut, needs to specifically pass each dimension,
    ie. (sidelen=[256,256]) for a square image, insetad of (sidelen=256, dim=2)
    Args
        sidelen     list, grid steps per dimension, tuples, ints or other iterables will fail
        ranges      list of ranges [None] | [[from:to(exlusive)], ....]

    Examples:
        >>> get_subgrid(sidelen=[121,34,34]) # return full meshgrid
        >>> get_subgrid(sidelen=[121,34,34], ranges=[[0,10]]) # slice in dim 0
        >>> get_subgrid(sidelen=[121,34,34], ranges=[[0,10],[3,4],[5,6]]) # slice all dims
        >>> get_subgrid(sidelen=[121,34,34], ranges=[None, None,[5,6]]) # slice last dim
    """
    if ranges is None:
        return _get_mgrid(sidelen)
    ranges, sublen,  = _check_ranges(sidelen, ranges)

    out = []
    for i, side in enumerate(sidelen):
        # grid step
        step = 2/(side - 1)
        # repreats
        pre = np.prod(sublen[:i]+ [1]) # prod(pre+post) = prod(sidelen)
        post = np.prod(sublen[i+1:] + [1])
        out += [(torch.arange(ranges[i][0], ranges[i][1])*step -1
                ).repeat_interleave(post).repeat(pre).view(-1,1)]
    return torch.cat(out, dim=1)

def _get_mgrid(sidelen):
    """  nd generic dim meshgrid
    Breaks get_mgrid sortcut, needs to specifically pass each dimension,
    ie. (sidelen=[256,256]) for a square image, insetad of (sidelen=256, dim2)
    Args
        sidelen     list , tuples, ints or other iterables will fail
    """
    # if subframes is not None:
    #     fro = subframes[0]*2/sidelen[0] -1
    #     to = subframes[0]*2/sidelen[0] -1
    out = []
    assert isinstance(sidelen, list), f"convert to list, {type(sidelen)} not supported"
    for i, side in enumerate(sidelen):
        pre = np.prod(sidelen[:i]+ [1]) # prod(pre+post) = prod(sidelen)
        post = np.prod(sidelen[i+1:] + [1])
        out += [torch.linspace(-1,1, side).repeat_interleave(post).repeat(pre).view(-1,1)]
    return torch.cat(out, dim=1)

def _check_ranges(sidelen, ranges):
    """ sanity check, range between 0,sidelen[i]
    """
    assert isinstance(sidelen, list), f"sidelen: list req, {type(sidelen)} not supported"

    sublen = []
    ranges = ranges if isinstance(ranges, list) else []
    for i in range( len(ranges), len(sidelen)):
        ranges += [[0, sidelen[i]]]

    for i, rng in enumerate(ranges):
        if not isinstance(rng, list) or not len(rng) == 2:
            ranges[i] = [0, sidelen[i]]
        else:
            ranges[i][0] = max(0, ranges[i][0])
            ranges[i][1] = min(ranges[i][1], sidelen[i])
            ranges[i][1] = max(ranges[i][1], ranges[i][0]+1)
        sublen += [ranges[i][1] - ranges[i][0]]
    return ranges, sublen

###
# Datasets
#
class VideoDataset(Dataset):
    """ VideoDataset
    # this is strange. each "epoch loads only 1 'sample' per epoch
    # with default sample size = 0.0038

    # Modified from dataset in github:
    * compute GPU utilization using x_utils.estimate_samples:
    given a video size it computes
    (num_params + num_latents for every computation) * byte_depth * 2 (gradients)
    including input and output

    gpu.available/gpu.required = sample size.

    CPU intilization, videos too large will swap.

    TODO test sequential, test image chunks,
    TODO use pseudorandom non repeating properties.

    currently:
    
    # CPU RAM is 3x the size of the video in float32 mean centered - in this version
    # 1x video frames
    # 1x mgrid
    # 1x shuffled indices

    index shuffling shouldnot utilize massive amount of ram.
    mgrid, can be made to load only requested indices,
    dataloader could be built to load random set of frames, iterate and continue
    
    Video Datastet, .npy, mp4, image foldr
        frame_range = only a few frames.
    Args
        path_to_video       (str) folder of images - req. images must be same size  TODO reenable mp4
        frame_range         (tuple, from to, [None]) fractional frame range.
        sample_fraction     (float [1]) fractional loader

    """
    def __init__(self, path_to_video, frame_range=None, sample_fraction=1., device="cpu", sample_size=None):
        super().__init__()
        # if 'npy' in path_to_video:
        #     self.vid = np.load(path_to_video)
        # elif 'mp4' in path_to_video:
        #     self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        if osp.isdir(path_to_video):
            open_tensor = lambda x, device="cuda": torch.as_tensor((np.asarray(Image.open(x), dtype=np.float32) - 127.5)/127.5, device=device)
            images = sorted([f.path for f in os.scandir(path_to_video)
                             if f.name[-4:].lower() in (".png", ".jpg", ".jpeg")])
            if frame_range is not None:
                print(f"frame_range, {frame_range[1]-frame_range[0]} from {len(images)} images")
                images = images[frame_range[0]:frame_range[1]]

            self.data = torch.stack([open_tensor(image, device) for image in images], dim=0)
            _shape = self.data.shape
            print(f"Loaded data, shape, {_shape}")
            self.data = self.data.view(-1, self.data.shape[-1])
            print(f" reshapesd shape, {self.data.shape}")

            self.channels = self.data.shape[-1]
        else:
            raise NotImplementedError("mp4, mov not yet implemented")
        print(f"   getting mgrid with sidelen {list(_shape[:-1])}")
        self.mgrid = get_mgrid(list(_shape[:-1]))

        if sample_size is not None:
            # drops last
            self.N_samples  = sample_size
            self.sample_fraction = min(1, sample_size / self.mgrid.shape[0])
        else:
            self.sample_fraction = sample_fraction
            self.N_samples = max(1, int(self.sample_fraction *  self.mgrid.shape[0]))
        print("   num samples, {}, fraction, {}, grid {}".format(self.N_samples, self.sample_fraction,
              tuple(self.mgrid.shape)))

        # should build a dataloader, or instantiate single coordinates.
        # if no connectivity is required, this could be a generated from
        # unique combination of sidelens in unseen locations : accumulate locations
        # maybe hash them.
        self.sampler = np.arange(self.mgrid.shape[0])
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.sampler)

    def __len__(self):
        # return 1
        return int(1/self.sample_fraction)

    def __getitem__(self, idx):
        """ 
        """
        if self.sample_fraction < 1.:
            # thats odd
            # some somples will be repeaated and some missed on every epoch, this way
            # coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))

            coord_idx = self.sampler[idx:idx+self.N_samples]
            data = self.data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = self.data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict
