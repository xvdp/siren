""" scripts leveraging opencv to load and crop video
"""
import shutil
from urllib.parse import urlparse
from io import BytesIO
import os
import os.path as osp
import requests
import numpy as np
import cv2
from PIL import Image
from x_log import logger

# pylint: disable=no-member
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
    log = logger("VideoDataset", level=loglevel)

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
