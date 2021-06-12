'''@x mod to train_video 
for experimentatio


'''

# Enable import from parent package
import sys
import os
import os.path as osp
import psutil

import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
# import skvideo.datasets

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import dataio, meta_modules, utils, training, loss_functions, modules
import x_utils



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

mem = []
if opt.verbose:
    gpu = x_utils.NvSMI('MB')
    cpu = x_utils.CPU('MB')
    print("GPU", gpu)
    print("CPU", cpu)
    mem.append({'gpu': gpu.available, "cpu":cpu.available})


vid_dataset = dataio.Video(opt.video_path)
if opt.verbose:
    cpu = x_utils.CPU('MB')
    gpu = x_utils.NvSMI('MB')
    _used = mem[-1]['cpu'] - cpu.available
    print("loaded video {} with shape {}, using {} MB,  {} MB remain".format(opt.video_path,
                                                              tuple(vid_dataset.vid.shape),
                                                              _used, cpu.available))
    mem.append({'gpu': gpu.available, "cpu":cpu.available})


coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape,
                                          sample_fraction=opt.sample_frac)
if opt.verbose:
    cpu = x_utils.CPU('MB')
    gpu = x_utils.NvSMI('MB')
    _used = mem[-1]['cpu'] - cpu.available

    print("wrapped dataset")
    print(f" mgrid {tuple(coord_dataset.mgrid.shape)}")
    print(f" data {tuple(coord_dataset.data.shape)}")
    print(" samples {}".format(coord_dataset.N_samples))
    print("video uses {} MB, {} MB remain".format(_used, cpu.available))
    mem.append({'gpu': gpu.available, "cpu":cpu.available})

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                        num_workers=0)
if opt.verbose:
    print(f"created dataloader len: {len(dataloader)}")

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=vid_dataset.channels,
                                 mode='mlp', hidden_features=1024, num_hidden_layers=3)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=vid_dataset.channels, mode=opt.model_type)
else:
    raise NotImplementedError

if opt.checkpoint_path is not None and osp.isdir(opt.checkpoint_path):
    model.load_state_dict(torch.load(opt.checkpoint_path))

model.cuda()
if opt.verbose:
    params = sum([p.numel() for p in model.parameters()])
    gpuse = torch.cuda.max_memory_reserved()/2**30
    cpu = x_utils.CPU('MB')
    gpu = x_utils.NvSMI('MB')
    _used = mem[-1]['cpu'] - cpu.available
    print("loaded model: {} parameters, torch uses {} GB".format(params, gpuse))

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_video_summary, vid_dataset)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
               verbose=opt.verbose)
