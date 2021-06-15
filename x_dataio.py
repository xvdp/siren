"""
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
from numpy.lib.arraysetops import isin
import torch
from torch.utils.data import Dataset


# pylint: disable=no-member
# pylint: disable=not-callable

def get_mgrid(sidelen):
    """  nd generic dim meshgrid
    Breaks get_mgrid sortcut, needs to specifically pass each dimension,
    ie. (sidelen=[256,256]) for a square image, insetad of (sidelen=256, dim2)
    Arg.
        sidelen     list , tuples, ints or other iterables will fail
    but still heavyish in that it is N*data size!!
    TODO: make functional - compute only positions that you need
    """
    out = []
    assert isinstance(sidelen, list), f"convert to list, {type(sidelen)} not supported"
    for i, side in enumerate(sidelen):
        pre = np.prod(sidelen[:i]+ [1]) # prod(pre+post) = prod(sidelen)
        post = np.prod(sidelen[i+1:] + [1])
        out += [torch.linspace(-1,1, side).repeat_interleave(post).repeat(pre).view(-1,1)]
    return torch.cat(out, dim=1)


class VideoDataset(Dataset):
    """ VideoDataset
    * mashup of dataio Video(Dataset) and Implicit3dWrapper - 3x less cpu ram required
    * can open directy to torch cuda - avoiding any cpu ops
    For small videos - TODO, chunked video datasets with cpu prefetch
    Video Datastet, .npy, mp4, image foldr
        frame_range = only a few frames.
    Args
        path_to_video       folder
        frame_range         (tuple, from to, [None]) fractional frame range.
        sample_fraction     (float [1]) fractional loader

    """
    def __init__(self, path_to_video, frame_range=None, sample_fraction=1., device="cpu"):
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
        self.sample_fraction = sample_fraction
        self.N_samples = max(1, int(self.sample_fraction * self.mgrid.shape[0]))
        print(f"   nuber of samples {self.N_samples}")


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """ TODO make functional, 
        """
        if self.sample_fraction < 1.:
            # thats odd
            # some somples will be repeaated and some missed on every epoch, this way
            # TODO use proper dataloader sampling. shuffle data and grab from index!!!
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
            data = self.data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = self.data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict
