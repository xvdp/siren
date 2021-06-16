import time
import os.path as osp
import numpy as np
import cv2
import torch
import modules
from x_dataio import get_mgrid

"""
TODO :
mgrid
    1. grid slicer, instead of loading an entire mgrid, create requrested indices only
model
    2. remvoe requires grad inside model.
training,
    3. save original frame and sizes to json or yaml
render
    4. evaluate max size of chunk that can be rendered at a time given GPU

"""
#model_output, _ = img_siren(get_mgrid(sz[0]).cuda())

#pylint: disable=no-member
def model_from_checkpoint(checkpoint):
    state = torch.load(checkpoint)

    count_params = [tuple(state[k].shape) for k in state if ".weight" in k]
    model_params = {"type":"sine",
                    "in_features":count_params[0][1],
                    "out_features":count_params[-1][0],
                    "mode":"mlp",
                    "hidden_features":count_params[0][0],
                    "num_hidden_layers":len(count_params) - 2
                    }
    channels = model_params["out_features"]

    model = modules.SingleBVPNet(**model_params)
    model.load_state_dict(state)
    return model, channels


def render_image(model, sidelen=[1426,256,256], frame_range=0):
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
        model, channels = model_from_checkpoint(model)
    else:
        channels = list(model.parameters())[-1].shape[0]

    # TODO make grid slicer
    if not isinstance(frame_range, (list, tuple)):
        frame_range = [frame_range, frame_range+1]
    mgrid = get_mgrid(sidelen)
    _from=np.prod([frame_range[0], *sidelen[1:]])
    _to=np.prod([frame_range[1], *sidelen[1:]])
    inputgrid = {"coords": mgrid[_from:_to]}

    output_size = [frame_range[1] -frame_range[0], *sidelen[1:], channels]

    with torch.no_grad():
        inputgrid["coords"] = inputgrid["coords"].cuda()
        model.cuda()
        model.eval()
        # TODO remove the "requires grad" in model definition
        out = model(inputgrid)["model_out"].cpu().detach().numpy() * 0.5 + 0.5

    del mgrid
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return out.reshape(*output_size)


def render_video(model, outname, sidelen=[1426,256,256], chunksize=12):#, show=True):
    """
    """
    _time = time.time()
    if isinstance(model, str) and osp.isfile(model):
        model, channels = model_from_checkpoint(model)
    else:
        channels = list(model.parameters())[-1].shape[0]


    # video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 29.97
    video = cv2.VideoWriter(outname, fourcc=fourcc, fps=fps, frameSize=sidelen[1:])

    # process video
    mgrid = get_mgrid(sidelen)

    from_frame = 0
    to_frame = from_frame + chunksize
    output_size = [chunksize, *sidelen[1:], channels]

    with torch.no_grad():
        model.cuda().eval()

        while to_frame <= sidelen[0]:
            to_frame = min(to_frame, sidelen[0])
            _from = np.prod([from_frame, *sidelen[1:]])
            _to = np.prod([to_frame, *sidelen[1:]])
    
            inputgrid = {"coords": mgrid[_from:_to]}
            inputgrid["coords"] = inputgrid["coords"].cuda()

            # this could probably be done asynchornously
            out = np.clip((model(inputgrid)["model_out"].cpu().detach().numpy() * 0.5 + 0.5), 0, 1.0)
            out = (out * 255).astype(np.uint8)
            for frame in out.reshape(*output_size):
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # if show:
                #     cv2.imshow('frame', frame)
            print(f"rendered: frames {from_frame}, {to_frame}      ", end="\r"  )
            
            from_frame = to_frame
            to_frame = to_frame + chunksize
    video.release()
    # cv2.destroyAllWindows()
    del mgrid
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _time = time.time() - _time

    print("\n total time, {}s, time/frame {}s".format(round(_time), round(_time/sidelen[0], 3)))

