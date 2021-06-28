import os
import os.path as osp
import torch
import x_dataio, x_utils
from x_training import train_video
from x_infer import render_video

# pylint: disable=no-member
def run_train(name, folder="x_periment_scripts", loop=0):
    if not osp.isfile(name):
        name = osp.join(folder, name)
    assert osp.isfile(name), f"experiment {name} does not exist"
    train_video(name, render={"loop":loop})


def render(max_frames=None, sidelen=[200,512,512]):
    # sidelen=[200,512,512]  # get from opt and original path
    #

    folder = "/media/z/Malatesta/zXb/share/siren/eclipse_512_sub"
    outname = osp.join(folder, "sidebyside.mp4")
    checkpoint = osp.join(folder, "model_final.pth")
    # outname = osp.join(folder, "sidebyside.mp4v")
    original_path = "/media/z/Malatesta/zXb/share/siren/data/eclipse_512"
    return render_video(checkpoint, outname, sidelen=sidelen, chunksize=3, original_path=original_path, max_frames=200)


def prepare_dset(name="xb_512.yml", **kwargs):
    folder = "x_periment_scripts" if "folder" not in kwargs else kwargs["folder"]
    t = torch.ones([1]).cuda()
    opt = x_utils.read_config(osp.join(folder, name))
    opt.update(**kwargs)
    gpus = x_utils.GPUse()
    opt.sample_size = x_utils.estimate_samples(gpus.available, **opt["siren"])*2
    return opt

def load_dset(name="xb_512.yml", **kwargs):
    opt = prepare_dset(name=name, **kwargs)

    return x_dataio.VideoDataset(opt.data_path, sample_size=opt.sample_size,
                                 frame_range=opt.frame_range, strategy=opt.strategy)


if __name__ == "__main__":

    ###
    # 10 frames using strategy 2 from a 100,512,512 video

    # run_train("xb_512_10f_s-1.yml")
    # -1 original - possibly repeating random
    # 1/5 dataset per (iter/epoch).
    # 1000 iters (~ 200x dset)      loss 0.0039


    # run_train("xb_512_10f_s1.yml")
    # 1 non repeatng shuffled, full data per epoch,
    # 5 iters per epoch
    # @ 1000 iters(~ 200x dset)     loss 0.0039
    # @ 5000 iters (1000x dset)     loss 0.0022

    # run_train("xb_512_10f_s2.yml") # high variance loss
    # 2 sparse and dense contiguous unshuffled
    # 16 iters per epoch, 2x dataset per epoch
    # 80 epochs ( 160x dataset) loss 0.005
    # 100 epochs( 200x dataset) loss 0.0041

    ###
    # 5 frames
    # run_train("xb_512_5f_s-1.yml")
    # 1000 iters(~ 300x dataset)    loss 0.0029

    ###
    # 100 frames
    # run_train("xb_512_100f_s1.yml")
    # stragegy 1, 58 iters per epoch
    # 

    # run_train("xb_512_100f_s-1.yml")
    # strategy -1, 1 iter per epoch, 1/58 dataset size per iter
    # 6000 epochs():

    # test cat
    # run_train("cat_s-1.yml", loop=1)
    # run_train("cat_s1.yml", loop=1)
    # run_train("cat_s2.yml", loop=1)
    run_train("cat_s-1_100k.yml")

    # NOT DONE - more capacity
    ###
    # 100 frames 2x siren
    # run_train("xb_512_100f_s1_2x.yml")
    # 1000 iters(~ 300x dataset):
