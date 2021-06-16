'''@x mod to train_video
for experimentatio


'''

# Enable import from parent package
import sys
import os
import os.path as osp
from functools import partial
import logging

import torch
from torch.utils.data import DataLoader
import configargparse
# import skvideo.datasets

_path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path not in sys.path:
    sys.path.append(_path)
import dataio, loss_functions, modules  # , meta_modules, utils, training
import x_utils
import x_training
import x_dataio


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=1000, # cat video does not need more than 1000 @x
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')



p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=38e-4,
               help='What fraction of video pixels to sample in each batch (default is all)')

# xvdp
p.add_argument('--video_path', default=None, required=True,  help='Path to Video') # x remove dataset add path to video
p.add_argument('--verbose', default=1,  help='log info')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.') # this argument is unused  @x

opt = p.parse_args()

# pylint: disable=no-member

_units = "GB"
M = x_utils.TraceMem(units=_units)
if opt.verbose:
    video_ram = x_utils.estimate_load_size(opt.video_path, units=_units, verbose=True)
    if 2*video_ram >= M.CPU[-1]:
        _ratio = M.CPU[-1]/video_ram
        raise ValueError(f"Video Too big, reduce or process in range chunks < {_ratio} of video")

# vid_dataset = dataio.Video(opt.video_path)
# if opt.verbose:
#     M.step(msg="Loaded Video")

# uses 2x the ram as the video
# TODO - optimize. coords -> function.
#                   data: deleete vid_dataset it wont be used.
# mgrid and

# channels = vid_dataset.channels
# coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape,
#                                           sample_fraction=opt.sample_frac)
# coord_dataset = dataio.CoordDataset3d(vid_dataset, sidelength=vid_dataset.shape,
#                                           sample_fraction=opt.sample_frac)

coord_dataset = x_dataio.VideoDataset(opt.video_path, frame_range=None, sample_fraction=opt.sample_frac)

print("coord shape", coord_dataset.data.shape)
print("op  sample_frac", opt.sample_frac)
print("coord sample_frac", coord_dataset.sample_fraction)
print("coord N_samples", coord_dataset.N_samples)
print("coord self.mgrid.shape[0]", coord_dataset.mgrid.shape[0])
channels = coord_dataset.channels


# del vid_dataset


M.step(msg="Coord Dataset", verbose=opt.verbose)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                        num_workers=0)
if opt.verbose:
    print(f"created dataloader len: {len(dataloader)}")

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=channels,
                                 mode='mlp', hidden_features=1024, num_hidden_layers=3)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=channels, mode=opt.model_type)
else:
    raise NotImplementedError
M.step(msg="model cpu", verbose=opt.verbose)

if opt.checkpoint_path is not None and osp.isdir(opt.checkpoint_path):
    model.load_state_dict(torch.load(opt.checkpoint_path))

model.cuda()
M.step(msg=f"model cuda, params {sum([p.numel() for p in model.parameters()])}", verbose=opt.verbose)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
# summary_fn = partial(utils.write_video_summary, vid_dataset)
#  summary_fn=summary_fn, remove tensorboard

M.log()
# try:
#     x_training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
#                 steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#                 model_dir=root_path, loss_fn=loss_fn,
#                 verbose=opt.verbose)
# except Exception as _e:
#     logging.exception( f"Premature interruption: {root_path}")
# finally:
#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()
