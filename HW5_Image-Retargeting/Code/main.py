""" 
CSCE  689 Image Retargeting
Assignment 5

@author: rizuj

"""

# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"

    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask


def SeamCarve(input, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 
    # 1)computing the energy function, 
    # 2) finding optimal seam, and 
    # 3) removing the seam. 
    # The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    ## seamcarve starts here
    input = np.float32(input)
    usemask = np.any(mask)

    usemask = False
    inSize = input.shape
    target_size   = (int(heightFac*inSize[0]), int(widthFac*inSize[1]))

    if widthFac == 1:
        # deal with the rotation
        seam_count = inSize[0] - target_size[0]
        #rotate back to get original
        input = np.rot90(input, 3)
        if usemask:
            mask = np.rot90(mask, 3)
    else:
        seam_count = inSize[1] - target_size[1]


    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    output = input.copy()

    for _ in range(0, seam_count):
        energy_img  = getImgEnergy(gray)

        if usemask:
            energy_img += mask * np.max(energy_img)

        seam_img = getBestSeam(energy_img)

        prev = gray.shape
        gray = gray[seam_img]
        gray.resize(prev[0], prev[1]-1)

        if usemask:
            mask_prev = mask.shape
            mask = mask[seam_img]
            mask.resize(mask_prev[0], mask_prev[1]-1)

        # create mask with same dimensions as image
        seam_3d = np.zeros_like(output)
        seam_3d = seam_3d.astype(bool)

        # copy your image_mask to all dimensions (i.e. colors) of your image
        for chnl in range(3):
            seam_3d[:, :, chnl] = seam_img.copy()

        prev_3d = output.shape
        output = output[seam_3d]
        output.resize(prev_3d[0], prev_3d[1]-1, prev_3d[2])

    if widthFac == 1:
        #rotate back to get original
        output = np.rot90(output)

    return output, target_size


def getImgEnergy(img_gray):

    energy = np.zeros(img_gray.shape)

    for i in range(0, img_gray.shape[0]):
        for j in range(0, img_gray.shape[1]):

            if i != 0:
                energy[i, j] += abs(img_gray[i, j] - img_gray[i-1, j])
            else:
                energy[i, j] += img_gray[i, j]

            if j != 0:
                energy[i, j] += abs(img_gray[i, j] - img_gray[i, j-1])
            else:
                energy[i, j] += img_gray[i, j]

    return energy


def getBestSeam(energy):

    seam = np.ones(energy.shape)
    seam = seam.astype(bool)

    dp_energy = energy.copy()

    for i in range(1, energy.shape[0]):
        for j in range(0, energy.shape[1]):

            if j == 0:
                #left border
                dp_energy[i, j] += min(dp_energy[i-1, j], dp_energy[i-1, j+1])
            elif j == energy.shape[1]-1:
                #right border
                dp_energy[i, j] += min(dp_energy[i-1, j-1], dp_energy[i-1, j])
            else:
                dp_energy[i, j] += min(dp_energy[i-1, j-1], dp_energy[i-1, j], dp_energy[i-1, j+1])


    min_idx = np.argmin(dp_energy[-1, :])
    seam[-1, min_idx] = False

    for idx in reversed(range(0, energy.shape[0]-1)):
        if min_idx == 0:
            new_idx = np.argmin(dp_energy[idx, min_idx:min_idx+2])
            min_idx += new_idx
        elif min_idx == energy.shape[1]-1:
            new_idx = np.argmin(dp_energy[idx, min_idx-1:min_idx+1])
            min_idx += new_idx-1
        else:
            new_idx = np.argmin(dp_energy[idx, min_idx-1:min_idx+2])
            min_idx += new_idx-1

        seam[idx, min_idx] = False

    return seam


# Driver Code

# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 1 # To reduce the width, set this parameter to a value less than 1
heightFac = 0.5  # To reduce the height, set this parameter to a value less than 1

N = 4 # number of images

for index in range(4, N + 1):

    input, mask = Read(str(index).zfill(2), inputDir)

    # Performing seam carving. This is the part that you have to implement.
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    # Writing the result
    plt.imsave("{}/w_result_nomask_{}_{}x{}.jpg".format(outputDir,
                                            str(index).zfill(2),
                                            str(size[0]).zfill(2),
                                            str(size[1]).zfill(2)), output)