'''Implements a generic training loop.
'''
import os
import os.path as osp
import time
import shutil
import logging
from x_infer import load_state_dict

import torch
from torch.utils.data import DataLoader
import pandas as pd # move to x_log

import vidi

import x_utils
import x_dataio
import x_modules
import x_infer
from x_log import PLog, sround


# remove tensorflue
# remove tqdm
# add panda log
# remove text loss

# pylint: disable=no-member
def _continue(folder, init=False, cleanup=False):
    """
    Create place holder file. Kill file to stop training.
    """
    _cont = osp.join(folder, "_continue_training")
    if init and not osp.isfile(_cont):
        with open(_cont, "w") as _fi:
            pass
    if osp.isfile(_cont):
        if cleanup:
            os.remove(_cont)
        else:
            return _cont
    return False

def load_last_checkpoint(folder, model):
    """ load last chechpoint
    """
    checkpoint = osp.join(folder, "model_final.pth")
    if not osp.isfile(checkpoint):
        checks = sorted([f.path for f in os.scandir(folder) if ".pth" in f.name])
        checkpoint = checks[-1] if checks else None

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        return checkpoint
    return None

def _prevent_overwrite(folder, model):
    """  if found, either overwrite or continue last checkpoint
    """
    epoch = 0
    total_time = 0
    if osp.exists(folder) and [f.name  for f in os.scandir(folder) if (".pth" in f.name or ".csv" in f.name)]:
        val = input("The model directory %s exists. Overwrite? (y/n)"%folder)
        if val == 'y':
            shutil.rmtree(folder)
        elif load_last_checkpoint(folder, model) is not None:

            # checkpoint = osp.join(folder, "model_final.pth")
            # if not osp.isfile(checkpoint):
            #     checks = sorted([f.path for f in os.scandir(folder) if ".pth" in f.name])
            #     checkpoint = checks[-1] if checks else None

            # if checkpoint is not None:
            #     state_dict = torch.load(checkpoint)
            #     model.load_state_dict(state_dict)

            plog = osp.join(folder, "train.csv")
            if osp.isfile(plog):
                df = pd.read_csv(plog)
                if "Epoch" in df:
                    epoch = int(df.Epoch.iloc[-1])
                if "Total_Time" in df:
                    total_time = df.Total_Time.iloc[-1]
    os.makedirs(folder, exist_ok=True)
    return epoch, total_time

def train(model, train_dataloader, epochs, lr, epochs_til_checkpoint, model_dir, dataset, **kwargs):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    # if use_lbfgs:
    #     optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
    #                               history_size=50, line_search_fn='strong_wolfe')
    num_steps = None if "num_steps" not in kwargs else kwargs["num_steps"]

    epoch_0, _extra_time = _prevent_overwrite(model_dir, model)

    log = PLog(osp.join(model_dir, "train.csv"))

    _continue(model_dir, init=True)
    total_steps = 0
    total_time = 0
    start_time = time.time()

    for epoch in range(epoch_0, epochs+epoch_0):
        # if epoch and shuffle_dataset:
        #     dataset.shuffle()

        if not _continue(model_dir) or (num_steps is not None and total_steps >= num_steps):
            break

        if not epoch % epochs_til_checkpoint and epoch:
            torch.save(model.state_dict(), osp.join(model_dir, 'model_epoch_%04d.pth' % epoch))

        for step, (model_input, gt) in enumerate(train_dataloader):
            log.collect(**{"Epoch":epoch, "Iter":step, "IterAll":total_steps})

            pos = model_input["coords"].cuda()
            target = gt["img"].cuda()

            log.collect(**{"tGPU":torch.cuda.max_memory_reserved()//2**20})
            log.collect(**{"GPU":x_utils.GPUse().used, "CPU":x_utils.CPUse().used})

            pred = model(pos)
            loss = ((target -pred)**2).mean()

            log.collect(**{"Loss":sround(loss.cpu().item(), 2)})

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_steps += 1

            total_time = time.time() - start_time + _extra_time
            _time = round(total_time/total_steps, 2)
            total_time = int(total_time)

            log.write(Time=_time, Total_Time=total_time)

        if dataset.strategy == 1:
            print(f"\nEnd of epoch {epoch}, iters {total_steps}: shuffle dataset")
            dataset.shuffle()

    checkpoint = osp.join(model_dir, 'model_final.pth')
    torch.save(model.state_dict(), checkpoint)
    print(f"\nFinished at epoch {epoch}, checkpoint{checkpoint}")
    _continue(model_dir, cleanup=True)

    return checkpoint


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


def train_video(config_file="experiment_scripts/x_eclipse_5122.yml", verbose=1, **kwargs):
    """  simple trainer
    kwargs
    shuffle_loader=None, shuffle_dataset=None,
    """
    try:
        ##
        # read config files
        opt = x_utils.read_config(config_file)
        opt.update(**kwargs)
        if "shuffle" not in opt:
            opt.shuffle = True
        if "strategy" not in opt:
            opt.strategy = 2
        if verbose: print(opt)

        ##
        # Estimate GPU Max allowed
        # 1. init cuda
        z = torch.ones([1]).cuda()
        gpus = x_utils.GPUse()
        if verbose: print(gpus)
        # 2. estimate max samples
        max_samples = x_utils.estimate_samples(gpus.available, **opt["siren"])
        if verbose: print(f"max samples {max_samples}*2 to fit within {gpus.available}MB")
        max_samples *= 2

        ##
        # dataset, loader, model
        if osp.splitext(opt.data_path)[1].lower() in (".mp4", ".avi", ".mov"):
            # print("datapath", opt.data_path
            opt.fps = vidi.ffprobe(opt.data_path)["avg_frame_rate"]

        dset = x_dataio.VideoDataset(opt.data_path, sample_size=max_samples, frame_range=opt.frame_range, strategy=opt.strategy)
        if verbose: print(f"Creating dataset of {len(dset)}, data loaded size {dset.data.shape}")
        dataloader = DataLoader(dset, shuffle=opt.shuffle, batch_size=1, pin_memory=True, num_workers=0)
        opt.sidelen = tuple(dset.data.shape[:-1])
        opt.chanels = dset.data.shape[-1]
        opt.sample_size = dset.sample_size
        opt.dset_len = len(dset)
        if "num_steps" not in opt:
            opt.num_steps = None

        # model
        model = x_modules.Siren(**opt["siren"])
        model.cuda()
        # optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())
        # loss_fn = lambda x,y: ((x-y)**2).mean()

        ##
        # logging
        folder = osp.join(opt.logging_root, opt.experiment_name)
        opt.to_yaml(osp.join(folder, "training_options.yml"))
        checkpoint = train(model, dataloader, opt.num_epochs, opt.lr, opt.epochs_til_checkpoint, folder, dataset=dset, num_steps=opt.num_steps)

        # cleanup
        sidelen = dset.sidelen.tolist()
        _cleanup(dset, dataloader)


        if "render" in kwargs:
            # option max_frames, original_path
            with torch.no_grad():
                model.eval()
                chunksize = int(x_utils.estimate_frames(sidelen[-2:], grad=0)//0.5)
                render = x_utils.EasyDict(model=model, sidelen=sidelen, chunksize=chunksize)
                if isinstance(kwargs["render"], dict):
                    render.update(kwargs["render"])
                if "name" not in render:
                    render.name = osp.join(folder, "infrerence_{:04}.mp4".format(opt.num_epochs))
                render.fps = opt.fps

                S = x_infer.SirenRender(**render)
                S.render_video()

            # x_infer.render_video(**render)

    except Exception as _e:
        logging.exception("train_video fails")

    finally:
        # cleanup
        _cleanup(dset, dataloader, model)

def _cleanup(*objs):
    for obj in objs:
        if obj is not None:
            del obj
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
