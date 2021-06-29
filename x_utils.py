"""
data utility functions
"""
from typing import Any
import subprocess as sp
import os
import os.path as osp
import shutil
from urllib.parse import non_hierarchical, urlparse
from io import BytesIO
import requests
import psutil

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import logging
import x_log


import hashlib
import json
# pylint: disable=no-member

###
# get image from url / file
#
def loadimg(url):
    """ load PIL.Image from url
    """
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    print("Code {} Failed to load <{}>".format(response.status_code, url))
    return None

def get_image(image, as_np=False, dtype=None):
    """ open image as PIL.Image or ndarray of dtype, or convert
    Args
        image   (path str | url str | np.ndarray | PIL.Image)
        as_np   (bool [False]) default: PIL.Image
        dtype   (str [None]) | 'uint8', |'float32', | 'float64'         
    """
    if isinstance(image, str):
        if osp.isfile(image):
            image = Image.open(image).convert('RGB')
        elif urlparse(image).scheme:
            image = loadimg(image)
        else:
            assert False, "invalid image {}".format(type(image))

    if isinstance(image, np.ndarray) and not as_np:
        if image.dtype != np.dtype('uint8'):
            image = (image*255).astype('uint8')
        image = Image.fromarray(image)
    elif as_np:
        image = _np_image(image, dtype=dtype)
    return image

def _np_image(image, dtype='float32'):
    """  convert to np.array to dtype
    Args
        dtype   (str ['float32']) | 'uint8', 'float64
    """
    image = np.asarray(image)
    if dtype is not None and image.dtype != np.dtype(dtype):
        # convert to uint.
        if dtype == 'uint8':
            if np.log2(image.max()) !=1: # float images with range up to 2
                image = image*255
        elif image.dtype == np.dtype('uint8'):
            image = image/255
        image = image.astype(dtype)
    return image


###
# get video from url or file
#
def get_video(url, tofolder="data", as_images=False, as_video=False, crop=False, max_size=None, show=True, frame_range=None, skip_frames=0):
    """ load video from folder or url
    default just plays it
    Args
        url         (str) url or filename
        tofolder    (str ['data']) if as_images or as_video, save to folder
        as_images   (bool [False]) save to pngs
        crop        (bool [False]) if True crop to square
        max_size    (tuple [None]) scale video
        skip_frames (int [0]) number of frames to skip for everyone saved

        # these options dont work in colab
        as_video    (bool [False]) save to local file if different from url
        show        (bool [True])
    Examples:
    # save video from url
    tofolder = "data"
    url = "https://images-assets.nasa.gov/video/AFRC-2017-11522-2_G-III_Eclipse_Totality/AFRC-2017-11522-2_G-III_Eclipse_Totality~orig.mp4"
    >>> get_video(url, as_video=True, tofolder=tofolder, show=True)

    # load from file save range to local cropped images
    url = "data/AFRC-2017-11522-2_G-III_Eclipse_Totality~orig.mp4"
    tofolder = "data/eclipse_512"
    >>> get_video(url, max_size=512, crop=True, frame_range=[6600, 8025], as_images=True, tofolder=tofolder, show=True, skip_frames=0)
    >>> get_video(name, tofolder="data/eclipse_fullsq", as_images=True, crop=True, frame_range=[6600, 8025])

    """
    # https://images.nasa.gov/
    #url = "http://images-assets.nasa.gov/video/jsc2017m000793_2017Eclipse_4K_YT/jsc2017m000793_2017Eclipse_4K_YT~orig.mp4"
    #url = "https://images-assets.nasa.gov/video/AFRC-2017-11522-2_G-III_Eclipse_Totality/AFRC-2017-11522-2_G-III_Eclipse_Totality~orig.mp4"
    # url = "https://images-assets.nasa.gov/video/GSFC_20160624_SDO_m12292_DoubleEclipse/GSFC_20160624_SDO_m12292_DoubleEclipse~orig.mp4"

    video = None
    out = None
    try:
        cv2.startWindowThread()
        video = cv2.VideoCapture(url)
        size = [int(video.get(3)), int(video.get(4))]

        _scale = 1
        if isinstance(max_size, (int)):
            max_size = [max_size, max_size]
        if isinstance(max_size, (list,tuple)):
            _xratio = size[0]/max_size[0]
            _yratio = size[1]/max_size[1]
            # if crop scale larger then crop
            _scale = max(_xratio, _yratio) if not crop else min(_xratio, _yratio)
            if _scale > 1:
                size[0] = int(size[0]//_scale)
                size[1] = int(size[1]//_scale)

        if crop:
            crop = min(size)

        size = tuple(size)
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        fps = video.get(cv2.CAP_PROP_FPS)
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(" size:  ", size)
        print(" fps:   ", fps)
        print(" fourcc:", fourcc)
        print(" frames:", count)

        i = 0
        if isinstance(frame_range, int):
            frame_range = (frame_range, count)
        if isinstance(frame_range, (list, tuple)):
            i = frame_range[0]
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])
            print(" range:", frame_range)


        outname = osp.basename(url)
        if tofolder is not None:
            os.makedirs(tofolder, exist_ok=True)
            outname = osp.join(tofolder, outname)
        if url == outname:
            as_video = False

        if as_video:
            _size = size if not crop else [crop, crop]
            out = cv2.VideoWriter(outname, fourcc=fourcc, fps=fps, frameSize=_size)
        if as_images:
            outname = osp.splitext(outname)[0]+"{:06d}.png"


        while True:
            r, frame = video.read()
            if frame is not None:
                if skip_frames and i%skip_frames:
                    i += 1
                    continue
                print("video frame: {:>6}  ".format(i), end="\r")
                if _scale > 1:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
                if crop:
                    _y, _x = frame.shape[:2]
                    _y = (_y - crop)//2
                    _x = (_x - crop)//2
                    frame = frame[_y:_y+crop, _x:_x+crop]

                if as_images:
                    name = outname.format(i)
                    if not osp.isfile(name):
                        # print("save image", name)
                        cv2.imwrite(name, frame)
                    # else:
                        # print("already exists", name)

                if as_video:
                    out.write(frame)

                if show:
                    cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if frame_range is not None and i >= frame_range[1]:
                    break
                i += 1

            else:
                print("done")
                break

    except:
        print(f"could not load url from {url}")

    for stream in [video, out]:
        if stream is not None:
            stream.release()

    cv2.destroyAllWindows()
    print(f"closed {url}")
    print(" frames:", count)
    print(" size:  ", size)
    print(" fps:   ", fps)
    print(" fourcc:", hex(fourcc))

def scale_min_side(image, size=512):
    """ rescale minimum side to size
    """
    _size = list(image.shape[:2])
    _i = np.argmax(_size)
    _large = max(_size)*size//min(_size)
    _size[1 - _i] = _large
    _size[_i] = size
    image = cv2.resize(image, _size, interpolation=cv2.INTER_CUBIC)
    return image


def preprocess_images(src, dst, size=512, crop=True, loglevel=20, force=False):
    """ scales and crops images in src folder saves in dst
        crop: square crop
    """
    log = x_log.logger("VideoDataset", level=loglevel)
    
    files = sorted([f.path for f in os.scandir(src) if osp.splitext(f.name)[1].lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        log[0].warning(f"no images found in {src}, nothing done")
        return
    os.makedirs(dst, exist_ok=True)
    if os.listdir(dst):
        if force:
            shutil.rmtree(dst)
            os.makedirs(dst)
        else:
            log[0].warning(f"not empty {dst}, nothing done")
            return
    log[1].terminator = "\r"
    for i, name in enumerate(files):
        log[0].debug(f"\n <{name}> is file {osp.isfile(name)}")
        image = cv2.imread(name)
        image = scale_min_side(image)

        if crop:
            _y, _x = image.shape[:2]
            _y = (_y - size)//2
            _x = (_x - size)//2
            image = image[_y:_y+size, _x:_x+size]

        dstname = osp.join(dst, osp.basename(name))
        cv2.imwrite(dstname, image)
        log[0].info(f" {i} of {len(files)} Saved image size {size} to {dstname} ")
    log[1].terminator = "\n"
    log[0].info(f"\nSaved {len(files)} files to {dst}")

# ###
# Memory management
#
class EasyDict(dict):
    """ dict with object access similar to code used by NVidia stylegan
        different from EasyDict in pypi
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def to_yaml(self, name):
        """ save to yaml"""
        os.makedirs(osp.split(name)[0], exist_ok=True)
        with open(name, 'w') as fi:
            yaml.dump(dict(self), fi)

class EasyTrace(EasyDict):
    """ dict with object access
        delta function for iterable members
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def delta(self, name, i=-1, j=-2):
        """ return self[name][i] - self[name][j]"""
        assert isinstance(self[name], (tuple, list, np.ndarray)), "cannot delta type {}".format(type(self[name]))
        assert abs(i) <= len(self[name]) and abs(j) <= len(self[name]), "indices ({} {}) outside of range {}".format(i, j, len(self[name]))
        return self[name][i] - self[name][j]

class TraceMem(EasyTrace):
    def __init__(self, units="MB"):
        self.units = units
        cpu = CPUse(units=self.units)
        gpu = GPUse(units=self.units)

        self.GPU = [gpu.available]
        self.CPU = [cpu.available]

        self.dGPU = [0]
        self.dCPU = [0]

        self.msg = ["Init"]
        self.log_mem(cpu, gpu)

    def log_mem(self, cpu, gpu):
        print(f"  CPU: avail: {cpu.available} {self.units} \tused: {cpu.used} {self.units} ({cpu.percent}%)")
        print(f"  GPU: avail: {gpu.available} {self.units} \tused: {gpu.used} {self.units} ({gpu.percent}%)")

    def step(self, msg="", i=-2, j=-1, verbose=True):
        cpu = CPUse(units=self.units)
        gpu = GPUse(units=self.units)
        self.CPU += [cpu.available]
        self.GPU += [gpu.available]
        self.msg +=[msg]
        dCPU = self.delta('CPU', i=i, j=j)
        dGPU = self.delta('GPU', i=i, j=j)
        self.dGPU += [dGPU]
        self.dCPU += [dCPU]

        if verbose:
            msg = msg + ": " if msg else ""

            print(f"{msg}Used CPU {dCPU}, GPU {dGPU} {self.units}")
            self.log_mem(cpu, gpu)
    def log(self):
        print("{:^6}{:>12}{:>12}{:>12}{:>12}".format("step", "CPU avail", "CPU added", "GPU avail", "GPU added"))
        for i in range(len(self.GPU)):
         print("{:^6}{:>12}{:>12}{:>12}{:>12}  {:<6}".format(i, f"{self.CPU[i]} {self.units}", f"({self.dCPU[i]})", f"{self.GPU[i]} {self.units}", f"({self.dGPU[i]})", self.msg[i]))


def get_smi(query):
    _cmd = ['nvidia-smi', '--query-gpu=memory.%s'%query, '--format=csv,nounits,noheader']
    return int(sp.check_output(_cmd, encoding='utf-8').split('\n')[0])

class GPUse:
    """wrap to nvidia-smi"""
    def __init__(self, units="MB"):
        self.total = get_smi("total")
        self.used = get_smi("used")
        self.available = self.total - self.used
        self.percent = round(100*self.used/self.total, 1)
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        if self.units[0].upper() == "G":
            self.units = "GB"
            self.total //= 2**10
            self.used //= 2**10
            self.available //= 2**10

    def __repr__(self):
        return "GPU: ({})".format(self.__dict__)

class CPUse:
    """ wrap to psutils to match NvSMI syntax"""
    def __init__(self, units="MB"):
        cpu = psutil.virtual_memory()
        self.total = cpu.total
        self.used = cpu.used
        self.available= cpu.available
        self.percent = cpu.percent
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        _scale = 20
        if self.units[0].upper() == "G":
            self.units = "GB"
            _scale = 30
        else:
            self.units = "MB"
        self.total //= 2**_scale
        self.used //= 2**_scale
        self.available //= 2**_scale

    def __repr__(self):
        return "CPU: ({})".format(self.__dict__)

def get_video_attrs(name):
    """get size info from video header as
        N, H, W, C
    """
    video = cv2.VideoCapture(name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    channels = int(video.get(cv2.CAP_PROP_CHANNEL))
    channels = channels if channels != 0 else 3
    video.release()
    # cv2.destroyAllWindows()
    return np.array([frames, height, width, channels])

def estimate_size(shape, dtype, units="MB"):
    scale = {"B":1, "KB":2**10, "MB":2**20, "GB":2**30}
    return np.product(np.asarray(shape)) * np.dtype(dtype).itemsize//scale[units]

def estimate_load_size(video, dtype="float32", units="MB", frame_range=None, grad=False, verbose=False):
    """ estimate size of a video or video folder
        Args
            video   (str) mp4, mov, image folder
            dtype   (str[float32])
            units   (str[MB]) | B | KB | MB | GB
            grad    (bool[False]) | if true double estimate
    """
    if osp.isdir(video):
        images = sorted([f.path for f in os.scandir(video)
                         if f.name[-4:].lower() in (".png", ".jpg", ".jpeg")])

        shape = np.array([len(images), *np.asarray(Image.open(images[0])).shape])
    else:
        shape = get_video_attrs(video)

    if isinstance(frame_range, (list,tuple)):
        frame_range = list(frame_range)
        frame_range[1] = min(frame_range[1], shape[0])
        shape[0] = frame_range[1] - frame_range[0]

    size = estimate_size(shape, dtype, units)
    if grad:
        size *=2
    if verbose:
        print("video with shape {}, requires {} {}".format(shape.tolist(), size, units))
    return size, shape.tolist()

def plot_mem(fname):
    """ plots graphs captuyred with ~/shh/memuse.sh
    """
    assert osp.isfile(fname), f"memory trace {fname} not found"
    mem = pd.read_csv(fname)
    gpu = np.asarray(mem.GPUse/1024)
    cpu = np.asarray(mem.CPUse/1024)

    yticks=[cpu[0], gpu[0], cpu[-1],gpu[-1], cpu.max(), gpu.max()]
    yticks = list(set([round(tick,2) for tick in yticks]))

    dCPU = cpu[-1] - cpu[0]
    dGPU = gpu[-1] - gpu[0]
    plt.plot(cpu, label=f"CPU {round(dCPU, 2)}")
    plt.plot(gpu, label=f"GPU {round(dGPU, 2)}")


    plt.yticks(yticks)

    plt.grid()
    plt.legend()
    plt.show()

###
#
# yaml config file utils
#
def get_abspath(fname):
    _fname = osp.abspath(osp.expanduser(fname))
    __fname = osp.abspath(osp.join(osp.split(__file__)[0], fname))

    for f in [_fname, __fname, fname]:
        if osp.exists(f):
            return f
    return fname

def read_config(config_file, defaults=None):
    """ reads yaml configs from expermients"""
    with open(config_file, "r") as _fi:
        data = yaml.load(_fi, Loader=yaml.FullLoader)

    opt = EasyDict()
    if isinstance(defaults, dict):
        opt.update(defaults)
    opt.update(data)
    for key in opt:
        if "path" in key and isinstance(opt[key], str):
            opt[key] = get_abspath(opt[key])
    for o in opt:
        if opt[o] == "None":
            opt[o] = None
    return opt

###
# rough memory estimate
#
def siren_latent_num(in_features=3, hidden_features=1024, out_features=3, hidden_layers=3,
                       include_bias=1, outermost_linear=1):
    """ number of operations in siren
    """
    latent_ops = 2 + include_bias # x=x*w, x=x+b, x=sin(x)
    features = in_features + out_features * (2-outermost_linear)
    features += (hidden_layers + 1) *  hidden_features * latent_ops
    return features

def siren_param_num(in_features=3, hidden_features=1024, out_features=3, hidden_layers=3):
    """ number of parameters of an even width MLP with weight and bias
    """
    num_params = (in_features + 1) * hidden_features # first: weights + bias
    num_params += (hidden_layers + 1) * (hidden_features + 1) * hidden_features # hidden
    num_params += (hidden_features + 1) * out_features # last
    return num_params

def estimate_siren_cost(samples, in_features=3, out_features=3, hidden_features=1024, hidden_layers=3,
                        byte_depth=4, grad=1, include_bias=1, outermost_linear=1, **kwargs):
    """
    estimates the GPU load required for siren
    """
    _bytes = byte_depth * (1 + grad)

    # layers params
    l_weights = siren_param_num(in_features, hidden_features, out_features,
                                hidden_layers) * _bytes

    # latent size
    features = siren_latent_num(in_features, hidden_features, out_features, hidden_layers,
                                include_bias, outermost_linear)

    l_latents = samples * features * _bytes

    return l_latents + l_weights

def estimate_samples(gpu_avail=None, units="MB", in_features=3, out_features=3,
                     hidden_features=1024, hidden_layers=3, outermost_linear=1,
                     byte_depth=4, grad=1, include_bias=1, include_model=1, **kwargs):
    """ NOTE::  operations are efficient multiply result * 2 for closer estimate

    estimates max number of samples that a siren model can take in - given a GPU budget

    Args
        gpu_avail       (int [None]) if None, compute
        units           (str ["MB"]) | GB
        in_features [3], out_features [3], hidden_features[1024], hidden_layers[3], outermost_linear[1]:
            ints, model definition
        byte_depth      (int [4]) assumes torch.float32
        grad            (int [1]) on inference grad=0
        include_bias    (int [1]) compute + bias as separate latent
        include_model   (bool [1]) | if model already loaded set to False

    Examples
    >>> grad = 1 # training
    >>> config = 'x_periment_scripts/eclipse_512_sub.yml'
    >>> opt = x_utils.read_config(config)
    >>> samples = x_utils.estimate_samples(x_utils.GPUse().available, grad=grad, **opt.siren)
    """
    if gpu_avail is None:
        gpu_avail = GPUse(units=units).available
    scale = {"KB":2**10, "MB":2**20, "GB":2**30}[units]
    _bytes = byte_depth * (1 + grad) / scale

    if include_model: # remove fixed cost of model
        l_weights = siren_param_num(in_features, hidden_features, out_features,
                                    hidden_layers) * _bytes
        gpu_avail -= l_weights

    features = siren_latent_num(in_features, hidden_features, out_features, hidden_layers,
                                include_bias, outermost_linear) * _bytes
    return int(gpu_avail // features)

def estimate_frames(frame_size, gpu_avail=None, units="MB", in_features=3, out_features=3,
                    hidden_features=1024, hidden_layers=3, outermost_linear=1,
                    byte_depth=4, grad=1, include_bias=1, include_model=1, **kwargs):
    """ NOTE::  operations are efficient multiply result * 2 for closer estimate
    estimates max number of frames that a siren model can take in - given a GPU budget
    Args
        frame_size  (list, tuple)
    other args from  estimate_samples()

    Examples: how many frame can be loaded in GPU for inference of sie 512,512
    >>> grad = 0 # inference
    >>> config = 'x_periment_scripts/eclipse_512_sub.yml'
    >>> opt = x_utils.read_config(config)
    >>> gpu_avail = x_utils.GPUse().available
    >>> frames = x_utils.estimate_frames(frame_size=[512,512], gpu_avail=gpu_avail, grad=grad, **opt.siren)

    # actual number of frames that can be loaded on inference
    >>>  n = int(x_utils.estimate_frames([512,512], grad=0)//0.5)
    """
    samples = estimate_samples(gpu_avail=gpu_avail, units=units, in_features=in_features,
                               out_features=out_features, hidden_features=hidden_features,
                               hidden_layers=hidden_layers, outermost_linear=outermost_linear,
                               byte_depth=byte_depth, grad=grad, include_bias=include_bias,
                               include_model=include_model, **kwargs)
    return samples / np.prod(frame_size)
