# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:22:15 2020
@author: rizuj
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


def get_view(src_img, block_size, view_point):

    if len(src_img.shape) != 3:
        print("Error with the image shape. Expecting 3D. Got image with shape {0}".format(
            src_img.shape))

    select_rows = np.mod(
        np.arange(src_img.shape[0]), block_size) == (view_point[0]-1)
    select_cols = np.mod(
        np.arange(src_img.shape[1]), block_size) == (view_point[1]-1)

    img_view = src_img[select_rows, :, :]
    img_view = img_view[:, select_cols, :]

    return img_view.copy()


# code starts here

images = ['LF_01.png', 'LF_02.png']
src_dir = "../Images/"
res_dir = "../Results/"

for image in images:
    lf_image = plt.imread(src_dir+image)
    block_size = 7

    # task1
    views = [(1, 1), (1, 7), (7, 1), (7, 7)]

    for view in views:
        view_img = get_view(lf_image, 7, view)
        view_name = "view_"+str(view[0])+"_"+str(view[1])+"_"
        plt.imsave(res_dir+view_name+image, view_img)

    # task 2
    apertures = [3, 2, 1, 0]

    for aper in apertures:
        aper_img = np.zeros(
            (int(lf_image.shape[0]/block_size), int(lf_image.shape[1]/block_size), 3))
        count = 0

        for i in range(-aper, aper+1):
            for j in range(-aper, aper+1):
                aper_img += get_view(lf_image, 7, (i+4, j+4))
                count += 1

        aper_img = aper_img / count
        aper_name = "aper_"+str((aper*2)+1)+"_"
        plt.imsave(res_dir+aper_name+image, aper_img)

    # task 3
    ds = range(-2, 3)

    for d in ds:
        img_re = np.zeros(
            (int(lf_image.shape[0]/block_size), int(lf_image.shape[1]/block_size), 3))

        for i in range(1, 8):
            for j in range(1, 8):
                img_temp = get_view(lf_image, 7, (i, j))
                di = (i-4) * d
                dj = (j-4) * d
                img_temp_roll = np.roll(img_temp, (di, dj), (0, 1))
                img_re += img_temp_roll

        img_re = img_re / 49
        re_name = "refocus_d"+str(d)+"_"
        plt.imsave(res_dir+re_name+image, img_re)
