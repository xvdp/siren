"""
"""
from posixpath import expanduser
import sys
from functools import partial
import logging
# import configargparse
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

_path =  osp.dirname(osp.dirname(osp.abspath(__file__)))
if _path not in sys.path:
    sys.path.append(_path)

import dataio
import loss_functions
import modules
import x_dataio
import x_utils
import x_training



# pylint: disable=no-member
def train_setup(video_path, logging_root, experiment_name, verbose=True, config="new", checkpoint_path=None,
                sample_frac=38e-4, batch_size=1, frame_range=None,
                lr=1e-4, num_epochs=1000, steps_til_summary=100, epochs_til_ckpt=500, **kwargs):

    _units = "GB"
    M = x_utils.TraceMem(units=_units)
    video_ram, shape = x_utils.estimate_load_size(video_path, units=_units, frame_range=frame_range, verbose=True)
    _buf = 4 # CPU RAM requires data + 3x data to load and create massive grid.
    # TODO hash or funcionalize grid on __getitem__()
    #
    if _buf*video_ram >= M.CPU[-1]:
        _ratio = M.CPU[-1]/(_buf*video_ram)
        _max_shape = [int(shape[0]*_ratio), *shape[1:]]
        _max_size = int(np.sqrt(np.prod(shape[1:3]) *_ratio))
        _max_size = [shape[0], _max_size, _max_size, *shape[3:]]
        raise ValueError(f"Video shape {shape} Too big, max allowed {_max_shape} or {_max_size}")

    if config in ("orig", "orig+"):
        vid_dataset = dataio.Video(video_path, frame_range=frame_range)
        M.step(msg="Loaded Video", verbose=verbose)

        # uses 2x the ram as the video
        # TODO - optimize. coords -> function.
        #                   data: deleete vid_dataset it wont be used.
        # mgrid and

        channels = vid_dataset.channels
        if config == "orig":
            coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape,
                                                     sample_fraction=sample_frac)
            M.step(msg="Coord Dataset", verbose=verbose)
        elif config == "orig+":
            coord_dataset = dataio.CoordDataset3d(vid_dataset, sidelength=vid_dataset.shape,
                                                  sample_fraction=sample_frac)
            M.step(msg="Coord Dataset", verbose=verbose)
            del vid_dataset
            M.step(msg="Coord Dataset2", verbose=verbose)
        
    else:

        #TODO compute sample fraction load
        # required: sample size * 1xgrid, 1xdata, 1xdatagrad

        # coord sample_frac 0.0038
        # coord N_samples 2193998
        # data.shape = [495, 1080, 1080]
        # self.N_samples = max(1, int(self.sample_fraction * np.prod([495, 1080, 1080]))

        # __getitem__() ->
        # coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
        # GPU LOAD: np.prod(coord_idx.shape) * np.dtype("float32").itemsize/2**30 * 3

        coord_dataset = x_dataio.VideoDataset(video_path, frame_range=frame_range, sample_fraction=sample_frac)
        channels = coord_dataset.channels

        M.step(msg="Coord Dataset", verbose=verbose)
            
        print("coord shape", coord_dataset.data.shape)
        print("op  sample_frac", sample_frac)
        print("coord sample_frac", coord_dataset.sample_fraction)
        print("coord N_samples", coord_dataset.N_samples)
        print("coord self.mgrid.shape[0]", coord_dataset.mgrid.shape[0])

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                        num_workers=0)
    if verbose:
        print(f"created dataloader len: {len(dataloader)}")

    model_type = "sine"
    model = modules.SingleBVPNet(type=model_type, in_features=3, out_features=channels,
                                 mode='mlp', hidden_features=1024, num_hidden_layers=3)
                        

    M.step(msg="model cpu", verbose=verbose)

    if checkpoint_path is not None and osp.isdir(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    model.cuda()
    M.step(msg=f"model cuda, params {sum([p.numel() for p in model.parameters()])}", verbose=verbose)

    # model ony requires 12MB  + 12MB gradients
    # sum([p.numel() for p in model.parameters()])*np.dtype("float32").itemsize/2**20
    # data asmpled only requires 25MB + 25MB coordinates ,,,, whty is it allocationg so much GPU?
    # self.N_samples * 3 * np.dtype("float32").itemsize/2**20
 


    root_path = osp.join(logging_root, experiment_name)

    # Define the loss
    loss_fn = partial(loss_functions.image_mse, None)


    M.log()

    return coord_dataset, dataloader, model

    # try:
    #     x_training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr,
    #                 steps_til_summary=steps_til_summary, epochs_til_checkpoint=epochs_til_ckpt,
    #                 model_dir=root_path, loss_fn=loss_fn,
    #                 verbose=verbose)
    # except Exception as _e:
    #     logging.exception( f"Premature interruption: {root_path}")
    # finally:
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

def read_config(config_file):
    with open(config_file, "r") as _fi:
        data = yaml.load(_fi, Loader=yaml.FullLoader)
    opt =  x_utils.EasyDict(data)
    opt.video_path = osp.abspath(osp.expanduser(opt.video_path))
    assert osp.exists(opt.video_path), f"file/ folder {opt.video_path} not found"
    for o in opt:
        if opt[o] == "None":
            opt[o] = None
    return opt
    

if __name__ == "__main__":

    assert len(sys.argv) > 1 and osp.isfile(sys.argv[1]), f"folder {sys.argv[1]} not found"

    OPT = read_config(sys.argv[1])
    #train_setup(opt)