'''Implements a generic training loop.
'''
import os
import os.path as osp
import time
import shutil

import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
import numpy as np


from x_utils import GPUse, CPUse
from x_log import PLog


# if opt.verbose:
#     params = sum([p.numel() for p in model.parameters()])
#     gpuse = torch.cuda.max_memory_reserved()/2**30
#     print("loaded model: {} parameters, {} GB VRAM".format(params, gpuse))

# remove tensorflue
# remove tqdm
# add panda log
# remove text loss

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

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          verbose=0):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    # if use_lbfgs:
    #     optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
    #                               history_size=50, line_search_fn='strong_wolfe')

    if osp.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    log = PLog(osp.join(model_dir, "train.csv"))
    vallog = PLog(osp.join(model_dir, "val.csv"))

    _continue(model_dir, init=True)
    total_steps = 0
    total_time = 0
    start_time = time.time()


    for epoch in range(epochs):
        if not _continue(model_dir):
            break

        if not epoch % epochs_til_checkpoint and epoch:
            torch.save(model.state_dict(), osp.join(model_dir, 'model_epoch_%04d.pth' % epoch))

        for step, (model_input, gt) in enumerate(train_dataloader):
            log.collect(**{"Epoch":epoch, "Step":total_steps})

            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            gpuse = torch.cuda.max_memory_reserved()/2**30
            log.collect(**{"tGPU":torch.cuda.max_memory_reserved()//2**20})
            log.collect(**{"GPU":GPUse().used, "CPU":CPUse().used})
            if verbose and total_steps == 0:
                mk = list(model_input.keys())
                gk = list(gt.keys())
                gpuse = torch.cuda.max_memory_reserved()/2**30
                print("step {}\t input_items {}\t item_size {}\t gt_size {}\t, gpu {} GB".format(step,  len(model_input.items()),
                                                                                        tuple(model_input[mk[0]].shape),
                                                                                        tuple(gt[gk[0]].shape), gpuse))

            # if double_precision:
            #     model_input = {key: value.double() for key, value in model_input.items()}
            #     gt = {key: value.double() for key, value in gt.items()}

            # if use_lbfgs:
            #     def closure():
            #         optim.zero_grad()
            #         model_output = model(model_input)
            #         losses = loss_fn(model_output, gt)
            #         train_loss = 0.
            #         for loss_name, loss in losses.items():
            #             train_loss += loss.mean()
            #         train_loss.backward()
            #         return train_loss
            #     optim.step(closure)

            model_output = model(model_input)
            losses = loss_fn(model_output, gt)
            if verbose and total_steps == 0:
                print("losses {}".format(len(losses)))

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()

                if loss_schedules is not None and loss_name in loss_schedules:
                    single_loss *= loss_schedules[loss_name](total_steps)

                train_loss += single_loss

            log.collect(**{"TrainLoss":train_loss.item()})

            if not total_steps % steps_til_summary:
                torch.save(model.state_dict(), osp.join(model_dir, 'model_current.pth'))

            # if not use_lbfgs:
            optim.zero_grad()
            train_loss.backward()

            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optim.step()

            if not total_steps % steps_til_summary:

                if val_dataloader is not None:
                    print("\nRunning validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_output = model(model_input)
                            val_loss = loss_fn(model_output, gt)
                            val_losses.append(val_loss.item())
                            vallog.write(**{"Epoch":epoch, "Step":total_steps, "Loss":val_loss.item() })

                    model.train()
            total_steps += 1

            total_time = time.time() - start_time
            _time = round(total_time/total_steps, 2)
            total_time = round(total_time, 1)

            log.write(Time=_time, Total_Tile=total_time)

    torch.save(model.state_dict(),osp.join(model_dir, 'model_final.pth'))

    _continue(model_dir, cleanup=True)


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
