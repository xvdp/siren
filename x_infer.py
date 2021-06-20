import time
import logging
import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import torch
import modules
from x_dataio import get_mgrid
from x_modules import Siren


try:
    # requires https:/github/xvdp/vidi
    import vidi
except: 
    print("To render video run\npython -m pip install git+https://github.com/xvdp/vidi")

"""
TODO :
mgrid
    1. grid slicer, instead of loading an entire mgrid, create requrested indices only
model
    2. remvoe requires grad inside model. # x_models SSiren
training,
    3. save original frame and sizes to json or yaml
render
    4. evaluate max size of chunk that can be rendered at a time given GPU

"""
#model_output, _ = img_siren(get_mgrid(sz[0]).cuda())

#pylint: disable=no-member
def model_from_checkpoint(checkpoint, simple_siren=True):
    state_dict = torch.load(checkpoint)

    count_params = [tuple(state_dict[k].shape) for k in state_dict if ".weight" in k]

    model_params = { "in_features":count_params[0][1],
                    "out_features":count_params[-1][0],
                    "hidden_features":count_params[0][0]}
    channels = model_params["out_features"]

    if simple_siren:
        model_params["hidden_layers"] = len(count_params) - 2
        model_params["outermost_linear"] = True
        model = Siren(**model_params)

    else:
        model_params["type"] = "sine"
        model_params["mode"] = "mlp"
        model_params["num_hidden_layers"] = len(count_params) - 2
        model = modules.SingleBVPNet(**model_params)

    assert load_state_dict(model, state_dict), "cannot load saved model"
    return model, channels

def load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    model_keys = list(model.state_dict().keys())
    load_keys = list(state_dict.keys())
    if load_keys == model_keys:
        model.load_state_dict(state_dict)
    else:
        for i, key in enumerate(model_keys):
            if model_state_dict[key].shape != state_dict[load_keys[i]].shape:
                print(key, "cannot be loaded from", load_keys[i])
                return False
            else:
                model_state_dict[key] = state_dict[load_keys[i]]

        model.load_state_dict(model_state_dict)
    return True

def render_image(model, sidelen=[1426,256,256], frame_range=0, simple_siren=True, device="cuda"):
    """ TODO on training save sidelen
        TODO evaluate max size of chunk that can be rendered at a time given GPU
    Args
        model with checkpoint, simply checkpoint
        sidelen     size of original siren
        frame_range
        checkpoint = "/media/z/Malatesta/zXb/share/siren/eclipse_256/model_final.pth"
        render_image(checkpoint, frame_range=(0,2))

        imgs = render_image(checkpoint, frame_range=[24,40]) # currently usess 16GB GPU
        measured with bash memuse.sh /media/z/Malatesta/zXb/share/siren/eclipse_256/render16.csv
        plot_mem("/media/z/Malatesta/zXb/share/siren/eclipse_256/render16.csv")

    """
    if isinstance(model, str) and osp.isfile(model):
        model, channels = model_from_checkpoint(model, simple_siren=simple_siren)
    else:
        channels = list(model.parameters())[-1].shape[0]

    # TODO make grid slicer
    if not isinstance(frame_range, (list, tuple)):
        frame_range = [frame_range, frame_range+1]
    output_size = [frame_range[1] -frame_range[0], *sidelen[1:], channels]

    # mgrid = get_mgrid(sidelen)
    # _from=np.prod([frame_range[0], *sidelen[1:]])
    # _to=np.prod([frame_range[1], *sidelen[1:]])

    with torch.no_grad():
        model.to(device=device).eval()

        # # coords = mgrid[_from:_to].to(device=device)
        coords = get_mgrid(sidelen, [frame_range]).to(device=device)
        if not simple_siren:
            out = model({"coords": coords})["model_out"]
        else:
            out = model(coords) # removes the "requires grad" in model definition
        out = out.cpu().detach().numpy() * 0.5 + 0.5

    del coords
    # del mgrid
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return out.reshape(*output_size)


# def load_siren(model, sidelen, simple_siren=True):
#     """
#     """
#     if isinstance(model, str) and osp.isfile(model):
#         model, channels = model_from_checkpoint(model)
#     else:
#         channels = list(model.parameters())[-1].shape[0]

# def render_frames(model, outname, dimensions, frames=None):
#     """
#         model       str checkpoint | nn.Module Siren
#         outname     str filename to render, extension determines output, None returns array
#         dimensions  list | tuple #equivalent to mgrid
#         frames      list, range [None] frames to rendeder
#     """

class SirenRender:
    """
    Args
        model       str checkpoint | nn.Module Siren
        name        str filename to render, extension determines output, None returns array
        sidelen    list | tuple # make mgrid
    kwargs
        frames      int, list, tuple, range
        
    """
    def __init__(self, model, name, sidelen, **kwargs):

        self.model = None
        self.channels = None
        self.name = name
        self.sidelen = sidelen

        self._output_type = "array"
        self.comp = None
        self._comp_type = None

        self._video = (".mp4", ".mov", ".avi")
        self._image = (".jpg", ".jpeg", ".png")


        self.set_output(self.name)
        self._get_model(model)

    def __del__(self):
        self.model = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _get_model(self, model):
        if isinstance(model, str) and osp.isfile(model):
            self.model, self.channels = model_from_checkpoint(model)
        else:
            self.channels = list(self.model.parameters())[-1].shape[0]

    def set_output(self, name):
        """  sets type and name of output
        Args
            name    None        ndarray
                    .png .jpg   image sequence
                    .mov .avi   video
        """
        self.name = name
        self._output_type = "array"

        if isinstance(name, str):
            self.name = osp.abspath(osp.expanduser(name))
            folder = osp.split(self.name)[0]
            os.makedirs(folder, exist_ok=True)
            ext = osp.splitext(self.name)[1].lower()
            if ext in self._image:
                self._output_type = "image"
            elif ext in self._video:
                self._output_type = "video"
            else:
                logging.warning(f"  Unsupported format {ext} not in {self._video} or {self._image}, dfaulting to 'array' output_type")

    def add_comp(self, data):
        """ add a comparison frames, resets the size of dimensions
            Args
                data    ndarray | folder | file list | file
        """
        _grid = None
        if data is None:
            self.comp = None
        else:
            if isinstance(data, np.ndarray):
                self.comp = data
                self._comp_type = "array"
                _ch = self.comp.shape[-1]
                _grid = self.comp.shape[:-1]

            elif isinstance(data, str) and osp.isdir(data):
                data = sorted([f.path for f in os.scandir(data) if osp.splitext(f.name)[1] in self._image])
                if not data:
                    logging.warning(f"  no images found in {data}, no image was added")
                else:
                    _sh = np.asarray(Image.open(data[0])).shape
                    _grid = [len(data), *_sh[:-1]]
                    _ch = _sh[-1]
                    self._comp_type = "images"
            elif isinstance(data, list) and osp.splitext(data[0])[1] in self._image:
                _sh = np.asarray(Image.open(data[0])).shape
                _grid = [len(data), *_sh[:-1]]
                _ch = _sh[-1]
                self._comp_type = "images"

            elif isinstance(data, str) and osp.isfile(data) and osp.splitext(data)[1] in self._video:
                _info = vidi.ffprobe(data)
                _grid = [_info["nb_frames"], _info["height"], _info['width']]
                _ch = 3 # not probing channels so this could fail
                self._comp_type = "video"

            if _grid is not None:
                if len(self.sidelen) > len(_grid):
                    _grid = [1] + _grid
                if len(self.sidelen) != len(_grid):
                    logging.warning(f"  invalid side len in comparision {_grid}, for {self.sidelen}")
                    self._comp_type = None
                if _ch != self.channels:
                    logging.warning(f"  invalid channels in comparison {_ch}, expected {self.channels}")
                    self._comp_type = None

            if self._comp_type is not None:
                self.comp = data
                self.sidelen = _grid
                logging.info(f"  added comp type{self._comp_type}, size {_grid}")

    def _addframes(self, frames):
        if isinstance(frames, int):
            frames = [frames]
        elif isinstance(frames, (list, tuple, range)):
            list(frames)
        else:
            frames = list(range(self.sidelen[0]))
        return frames

    def render_video(self, frames=None, chunksize=3):
        # TODO add chunksize estimatio
        # TODO hypernets


        frames = self._addframes(frames)
        frame_size = self.sidelen[1:] if self._comp_type is None else self.sidelen[1:]*2

        output_size = [chunksize, *self.sidelen[1:], self.channels]

        with vidi.FFcap(self.name, size=frame_size, fps=30) as Vidi:
            with torch.no_grad():
                self.model.cuda().eval()
                while frames:
                    chunksize = min(chunksize, len(frames))
                    output_size = [chunksize, *self.sidelen[1:], self.channels]
                    if frames[chunksize-1] - frames[0] == chunksize - 1: # frames are contiguous
                        coords = get_mgrid(self.sidelen, [[frames[0], frames[chunksize]]]).cuda()
                    else:
                        coords = torch.cat([get_mgrid(self.sidelen, [[frames[i], frames[i+1]]])
                                            for i in range(chunksize)]).cuda()

                    for _ in range(chunksize):
                        frames.pop(0)

                    out = self.model(coords)
                    out = np.clip((out.cpu().detach().numpy() * 0.5 + 0.5), 0, 1.0)
                    out = (out * 255).astype(np.uint8).reshape(chunksize, *output_size[1:])

                    for i, frame in enumerate(out):
                        Vidi.add_frame(frame)
        del coords
        del out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def play(self):
        vidi.ffplay(self.name)


def render_video(model, outname, sidelen=[1426,256,256], chunksize=12, simple_siren=True, original_path=None, max_frames=None):
    """
    Args
        model           (nn.Module, str) siren model or chekckpoint path
        outname         (str)   path to save
        sidelen         (list)  grid size N,H,W - could be same or different than original
        chunksize       (int)   number of frames rendered per iteration # no RAM check yet, TODO
        simple_siren    (bool) True, use x_module.Siren
        original_path   (str [None]) if video or path to original video, render side by side
        # original_path overrides sidelen

    TODO flip around, if not original_path, require sidelen
    if sidelen render only n frames from original path, maybe add skip
    TODO implement better control of sparsity
    """
    _time = time.time()
    if isinstance(model, str) and osp.isfile(model):
        model, channels = model_from_checkpoint(model)
    else:
        channels = list(model.parameters())[-1].shape[0]

    frame_size = sidelen[1:]
    if original_path is not None:
        assert osp.exists(original_path), f"original video could not be found in {original_path}"
        image_files = sorted([f.path for f in os.scandir(original_path)
                             if f.name[-4:].lower() in (".png", ".jpg", ".jpeg")])
        assert len(image_files), f"no images found {original_path}"

        sidelen = [len(image_files), *Image.open(image_files[0]).size]
        frame_size = sidelen[1:]
        frame_size[-1] *= 2

    from_frame = 0
    to_frame = from_frame + chunksize
    output_size = [chunksize, *sidelen[1:], channels]
    print(output_size)

    max_frames = sidelen[0] if max_frames is None else min(sidelen[0], max_frames)

    with vidi.FFcap(outname, size=frame_size, fps=30) as Vidi:
        with torch.no_grad():
            model.cuda().eval()

            while from_frame < max_frames:
                to_frame = min(to_frame, max_frames)

                _num = to_frame - from_frame    # number of frames to render at one time

                # sub grid for these frames
                coords = get_mgrid(sidelen, [[from_frame, to_frame]]).cuda()
                if not simple_siren:
                    out = model({"coords": coords})["model_out"]
                else:
                    out = model(coords) # removes the "requires grad" in model definition
                out = np.clip((out.cpu().detach().numpy() * 0.5 + 0.5), 0, 1.0)
                out = (out * 255).astype(np.uint8).reshape(_num, *output_size[1:])

                for i, frame in enumerate(out):
                    if original_path is not None:
                        image = np.asarray(Image.open(image_files[from_frame + i]))
                        frame = np.concatenate([image, frame], axis=1)
                    Vidi.add_frame(frame)

                    # plt.imshow(frame)
                    # plt.show()
                    # return


                print(f"rendered: frames {from_frame}, {to_frame}      ", end="\r")
                from_frame = to_frame
                to_frame = to_frame + chunksize

    del coords
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _time = time.time() - _time

    print("\nSaved to {} total time, {}s, time/frame {}s".format(outname, round(_time), round(_time/sidelen[0], 3)))

    vidi.ffplay(outname)

# def test_vid(max_frames=None):
#     #from x_infer import render_siren_video
#     # import os.path as osp

#     folder = "/media/z/Malatesta/zXb/share/siren/eclipse_5122"
#     outname = osp.join(folder, "sidebyside.mp4")
#     # outname = osp.join(folder, "sidebyside.mp4v")
#     original_path = "/media/z/Malatesta/zXb/share/siren/data/eclipse_512"
#     return render_siren_video(checkpoint, outname, chunksize=3, original_path=original_path, max_frames=max_frames)

