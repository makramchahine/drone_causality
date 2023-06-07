import json
import os
from pathlib import Path
from typing import Optional, Callable, Sequence, Union, Dict, Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor
from tensorflow.python.keras.models import Functional
from tqdm import tqdm
from skimage.metrics import structural_similarity

from utils.polygooner import PolygonDrawer, FINAL_LINE_COLOR, PolyArea
from keras_models import IMAGE_SHAPE, IMAGE_SHAPE_CV
from utils.data_utils import image_dir_generator
from utils.model_utils import generate_hidden_list, NCPParams, LSTMParams, CTRNNParams, TCNParams

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utils.triche import POLY_TRICHE, POLY_RECT, POLY_PATIO

TEXT_BOX_HEIGHT = 30

def run_attention(polygon, vis_model: Functional, data: Union[str, Iterable], vis_func: Callable,
                      image_output_path: Optional[str] = None,
                      video_output_path: Optional[str] = None, reverse_channels: bool = True,
                      control_source: Union[str, Functional, None] = None, absolute_norm: bool = True,
                      vis_kwargs: Optional[Dict[str, Any]] = None) -> Sequence[ndarray]:
    """
    Runner script that loads images, runs VisualBackProp, and saves saliency maps
    """
    if vis_kwargs is None:
        vis_kwargs = {}

    if isinstance(data, str):
        data = image_dir_generator(data, IMAGE_SHAPE, reverse_channels)

    # create output_dir if not present
    # if image_output_path is not None:
    #     Path(image_output_path).mkdir(parents=True, exist_ok=True)
    if video_output_path is not None:
        Path(os.path.dirname(video_output_path)).mkdir(parents=True, exist_ok=True)

    if len(vis_model.inputs) > 1:
        vis_hiddens = generate_hidden_list(vis_model, False)
    else:
        # vis_model doesn't need hidden state
        vis_hiddens = [None]

    if isinstance(control_source, Functional):
        control_hiddens = generate_hidden_list(control_source, True)
    elif isinstance(control_source, str):
        control_source = pd.read_csv(control_source)

    saliency_imgs = []
    og_imgs = []
    extra_imgs = []
    controls = []
    csv_healthy = True
    insid = []
    intns = []
    tit = []
    nap = []
    chairea = []
    ssim = []
    ROC = []
    for i, img in tqdm(enumerate(data)):
        og_imgs.append(img)
        saliency, vis_hiddens, sample_extra = vis_func(img, vis_model, vis_hiddens, **vis_kwargs)
        saliency_imgs.append(saliency)
        extra_imgs.append(sample_extra)

        if control_source is not None:
            if isinstance(control_source, Functional):
                out = control_source.predict([img, *control_hiddens])
                vel_cmd = out[0]
                control_hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim
            elif isinstance(control_source, DataFrame):
                try:
                    vel_cmd = np.nan_to_num(
                        control_source.iloc[i][["cmd_vx", "cmd_vy", "cmd_vz", "cmd_omega"]].to_numpy())
                except IndexError:
                    vel_cmd = np.zeros((4,))
                    if csv_healthy:
                        # log error
                        csv_healthy = False
                        csv_rows = control_source.shape[0]
                        image_num = len([c for c in os.listdir(data) if 'png' in c])
                        print(f"Warning: CSV for {data} has {csv_rows} rows and {image_num} images")

                vel_cmd = np.expand_dims(vel_cmd, axis=0)
            else:
                raise ValueError(f"Unsupported control source {control_source}")
            controls.append(vel_cmd)

    # normalize and display saliency images
    video_frames = []
    data_list = zip(range(len(saliency_imgs)), og_imgs, saliency_imgs, extra_imgs,
                    controls if controls else [None] * len(saliency_imgs))

    # calculate absolute min and max
    saliency_min = None
    saliency_max = None
    if absolute_norm:
        saliency_ndarr = np.asarray(saliency_imgs)
        saliency_min = np.min(saliency_ndarr)
        saliency_max = np.max(saliency_ndarr)
    # prepare video frames
    saliency_written = []
    for i, img, saliency, extra, vel_cmd in data_list:
        saliency_writeable = convert_to_color_frame(saliency, desired_size=IMAGE_SHAPE_CV, min_value=saliency_min,
                                          max_value=saliency_max)
        saliency_map = np.array(cv2.cvtColor(saliency_writeable[..., ::-1], cv2.COLOR_BGR2GRAY))
        # ret, saliency_map = cv2.threshold(saliency_map, 50, 255, cv2.THRESH_TRUNC)
        # nap.append(len(cv2.findNonZero(saliency_map)))
        # insides = count_in_polygon(saliency_map, np.array([polygon[i].points]))
        # insid.append(insides)
        # chairea.append(polygon[i].area)
        ssim.append(struct_sim(saliency_map, np.array([polygon[i].points]), image_output_path, i))
        roc, etalon = AUC_Judd(saliency_map, polygon[i], 200)
        ROC.append(roc)
        # intens, total_intens = intens_in_polygon(saliency, np.array([polygon[i].points]))
        # intns.append(intens)
        # tit.append(total_intens)

        saliency_written.append(saliency_writeable)
        saliency_writeable = np.ascontiguousarray(saliency_writeable, dtype=np.uint8)
        saliency = np.repeat(saliency, 3, axis=-1)
        saliency_map = np.expand_dims(saliency_map, axis=-1)
        saliency_map = np.repeat(saliency_map, 3, axis=-1)
        saliency_map = saliency_map

        # txt_dsp = "SCORE = "+str(round(insid[i] * 100, 1))+"% , NUM_ACT_PIX = "+str(nap[i])
        # cv2.putText(saliency_writeable, txt_dsp, (0, TEXT_BOX_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)


        ogim = np.squeeze(img)[..., ::-1]
        # cv2.imshow('og', np.squeeze(img)[..., ::-1])
        # cv2.imshow('saliency', saliency)
        # cv2.imshow('saliency_map', saliency_map)
        # cv2.imshow('saliency_writeable', saliency_writeable)


        sal_raw = np.squeeze(np.array([saliency_imgs[i]]), axis=0)
        sal_raw = cv2.normalize(sal_raw, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
        sal_raw = cv2.resize(sal_raw,IMAGE_SHAPE_CV,fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        sal_raw = np.expand_dims(sal_raw, axis=-1)
        sal_raw = np.repeat(sal_raw, 3, axis=-1)
        cv2.polylines(sal_raw, np.array([polygon[i].points]), True, FINAL_LINE_COLOR, 1)
        cv2.polylines(saliency_map, np.array([polygon[i].points]), True, FINAL_LINE_COLOR, 1)
        stack1 = np.concatenate([ogim, sal_raw], axis=0)
        stack2 = np.concatenate([saliency_writeable, saliency_map], axis=0)
        stackf = np.concatenate([stack1, stack2], axis=1)
        video_frames.append(stackf)
        file = image_output_path + f"saliency_mask_{i}.png"
        cv2.imwrite(file, stackf)

        cv2.imshow("wel3a", stackf)
        cv2.waitKey(1)
    # write video
    # if video_output_path:
    #     write_video(img_seq=video_frames, output_path=video_output_path)

    # d = {'chairea':chairea, 'num_act_pix':nap, 'num_in_poly':insid, 'total_att':tit, 'att_in_poly':intns, 'ssim':ssim, 'ROC':ROC}
    d = {'ssim':ssim, 'ROC':ROC}
    df = pd.DataFrame(data=d)
    ceesve = image_output_path+"numbers.csv"
    df.to_csv(ceesve, index=False)
    return saliency_written



def hand_annotate(data: Union[str, Iterable],reverse_channels: bool = True):
    if isinstance(data, str):
        data = image_dir_generator(data, IMAGE_SHAPE, reverse_channels)
    polygons = []
    for i, img in tqdm(enumerate(data)):
        img = img[..., ::-1]
        imag = np.ascontiguousarray(img, dtype=np.uint8)
        polyd = PolygonDrawer("Polygon"+str(i), imag[0])
        # canvas = polyd.run()
        polyd.points = POLY_PATIO[i]
        polyd.area = PolyArea(np.array(polyd.points))
        polygons.append(polyd)
    return polygons


def parse_params_json(params_path: str, set_single_step: bool = True):
    with open(params_path, "r") as f:
        params_data = json.loads(f.read())

    for local_path, params_str in params_data.items():
        model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams, None] = eval(params_str)
        if set_single_step:
            model_params.single_step = True
        model_path = os.path.join(os.path.dirname(params_path), local_path)
        yield local_path, model_path, model_params


def convert_to_color_frame(saliency_map: Union[Tensor, ndarray], desired_size: Optional[Sequence[int]] = None,
                           min_value: Optional[float] = None, max_value: Optional[float] = None,
                           color_map: int = cv2.COLORMAP_INFERNO) -> ndarray:
    """
    Converts tensorflow tensor (1 channel) to 3-channel grayscale numpy array for use with OpenCV
    """
    assert (min_value is None) == (max_value is None), "Pass both min and max or neither"
    if isinstance(saliency_map, Tensor):
        saliency_map = saliency_map.numpy()
    # add dummy color channel if not present
    if len(saliency_map.shape) == 2:
        saliency_map = np.expand_dims(saliency_map, axis=-1)

    # if grayscale image, repeat to virtually make color
    if saliency_map.shape[-1] == 1:
        saliency_map = np.repeat(saliency_map, 3, axis=-1)

    # resize to desired size if specified
    if desired_size is not None:
        saliency_map = cv2.resize(saliency_map, desired_size, )

    if min_value is not None and max_value is not None:
        saliency_map = saliency_map - min_value
        saliency_map = saliency_map / (max_value - min_value) * 255
        saliency_map = saliency_map.astype(np.uint8)
    else:
        saliency_map = cv2.normalize(saliency_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)

    # Thresh the bits out of it
    # ret, saliency_map = cv2.threshold(saliency_map, 10, 255, cv2.THRESH_BINARY)
    saliency_map = cv2.applyColorMap(saliency_map, color_map)#[..., ::-1]
    return saliency_map

def count_in_polygon(saliency, polygon):
    insides = 0
    # height, width = saliency.shape
    # blank_image = np.zeros((height, width, 3), np.uint8)
    pts = cv2.findNonZero(saliency)
    poly = Polygon(polygon[0])
    for i in range(0, len(pts)):
        ali = tuple(map(tuple, pts[i]))[0]
        alo = Point(ali[0].item(), ali[1].item())
        if poly.contains(alo):
            insides += 1
    return insides

def AUC_Judd(saliency, polygon, precision):
    ROC = np.zeros((precision,2))
    # add dummy color channel if not present
    # if len(saliency.shape) == 2:
    #     saliency = np.expand_dims(saliency, axis=-1)
    #
    # # if grayscale image, repeat to virtually make color
    # if saliency.shape[-1] == 1:
    #     saliency = np.repeat(saliency, 3, axis=-1)

    height, width = saliency.shape

    thrash = np.linspace(0,255, precision)
    poly = Polygon(np.array(polygon.points))
    etalon = np.zeros_like(saliency)
    cv2.fillPoly(etalon, np.array([polygon.points]), 255)
    ptrue = cv2.findNonZero(etalon)
    rng = cv2.countNonZero(etalon)
    B = np.squeeze(ptrue)

    for i in range(precision):
        ret, saliency_map = cv2.threshold(saliency+0.01, thrash[i], 255, cv2.THRESH_BINARY)
        ptsal = cv2.findNonZero(saliency_map)
        att = cv2.countNonZero(saliency_map)
        # cv2.imshow('img', saliency_map)
        # cv2.waitKey(0)
        # print(ptsal)
        try:
            A = np.squeeze(ptsal)
            nrows, ncols = A.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [A.dtype]}

            C = np.intersect1d(A.view(dtype), B.view(dtype))
            C = C.view(A.dtype).reshape(-1, ncols)
            green, zebb = C.shape
            red = rng - green
            blue = att - green
            yellow = height * width - att - red

            TP = green / (green + red)
            FP = blue / (blue + yellow)
        except:
            TP=0.0
            FP=0.0

        ROC[i, :] = [TP, FP]

        # cv2.imshow('etal', etalon)
        # cv2.imshow('img', saliency_map)
        # cv2.waitKey(1)

    return ROC, etalon

def intens_in_polygon(saliency, polygon):
    intens = 0
    tit = 0
    # height, width = saliency.shape
    # blank_image = np.zeros((height, width, 3), np.uint8)
    pts = cv2.findNonZero(saliency)
    poly = Polygon(polygon[0])
    for i in range(0, len(pts)):
        ali = tuple(map(tuple, pts[i]))[0]
        alo = Point(ali[0].item(), ali[1].item())
        # blank_image = cv2.circle(blank_image, alo, radius=0, color=(0, 0, 255), thickness=-1)
        # if cv2.pointPolygonTest(polygon, alo, False) ==1:
        #     insides +=1
        # elif cv2.pointPolygonTest(polygon, alo, False) ==0:
        #     insides += 1
        if poly.contains(alo):
            intens += saliency[ali[1], ali[0], 0]
        tit += saliency[ali[1], ali[0], 0]
    return intens, tit

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""

    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j],dtype=object)
            count += 1
    return sig

def struct_sim(saliency, polygon, image_output_path, i):
    etalon = np.zeros_like(saliency)
    cv2.fillPoly(etalon, pts = polygon, color=(255,255,255))
    file = image_output_path + f"ground_truth_{i}.png"
    # cv2.imwrite(file, etalon)
    (score, diff) = structural_similarity(np.squeeze(saliency), np.squeeze(etalon), full=True)
    # dist, _, flow = cv2.EMD(img_to_sig(saliency),img_to_sig(etalon),cv2.DIST_L2)
    return score



def write_video(img_seq: Sequence[ndarray], output_path: str, fps: int = 10):
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    seq_shapes = [img.shape for img in img_seq]
    assert seq_shapes.count(seq_shapes[0]) == len(seq_shapes), "Not all shapes in img_seq are the same"

    image_shape = img_seq[0].shape
    cv_shape = (image_shape[1], image_shape[0])  # videowriter takes width, height, image_shape is height, width
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, cv_shape,
                             True)  # true means write color frames

    for img in img_seq:
        writer.write(img)

    writer.release()