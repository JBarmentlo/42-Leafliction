from typing import List
from plantcv import plantcv as pcv
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch import Tensor
from PIL import Image
import torch
import cv2

from .types import rgb_image, rgb_image_cv, mask

def tensor_to_cv(im: rgb_image) -> rgb_image_cv:
    return np.array(to_pil_image(im))[:,:,::-1]

def cv_to_tensor(im: rgb_image_cv) -> rgb_image:
    return to_tensor(Image.fromarray(im))

def get_mask(im: rgb_image) -> mask:
    im_cv = tensor_to_cv(im)
    # im_last = im.transpose(0, 1).transpose(1, 2)
    thresh1 = pcv.threshold.dual_channels(rgb_img = im_cv, x_channel = "a", y_channel = "b", points = [(80,80),(125,140)], above=True)
    thresh1 = pcv.fill_holes(thresh1)
    return Tensor(thresh1 / 255.)

def transform_image(im: rgb_image) -> List[rgb_image]:
    cv_im = tensor_to_cv(im)
    mask = get_mask(im)
    shape_image = cv_to_tensor(pcv.analyze.size(img=cv_im, labeled_mask=mask.to(torch.uint8).numpy() * 255, n_labels=1))
    gaussian_img = cv_to_tensor(pcv.gaussian_blur(img=tensor_to_cv(im), ksize=(5, 5), sigma_x=0, sigma_y=None))
    homolog_pts, start_pts, stop_pts, ptvals, chain, max_dist = pcv.homology.acute(img=cv_im, mask=mask.to(torch.uint8).numpy() * 255, win=25, threshold=90)

    imm = cv_im
    for point in homolog_pts:
        imm = cv2.circle(imm, point[0], 1, (255, 0, 0), 2)
    landmarks = cv_to_tensor(imm)
    color_histogram, _ = pcv.visualize.histogram(img=tensor_to_cv(im), mask=mask, hist_data=True, bins=30)
    
    
    return [mask, mask * im, shape_image, gaussian_img, landmarks, color_histogram]

def do_the_thing(paf):
    im = to_tensor(Image.open(paf))
    images = transform_image(im)
    
    