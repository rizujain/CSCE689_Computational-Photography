"""
Assignment 3

@author: rizuj
@name: Rizu Jain
@UIN: 430000753

"""

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2


from datetime import datetime

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg") / 255
    target = plt.imread(path + "target_" + id + ".jpg") / 255
    mask   = plt.imread(path + "mask_" + id + ".jpg") / 255

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset

    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]

    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal


def myUpsample(img_in):

    img_out = cv2.pyrUp(img_in)
    return img_out


def myDownsample(img_in):

    img_out = cv2.pyrDown(img_in)
    return img_out


def gauss_pyramid(img_in, numLevels):

    # Create a list that will contain all the samples of the img_in.
    pyramid = [img_in]

    for l in range(numLevels):
        # Downsample the last level of image
        reduced = myDownsample(pyramid[-1])
        # Append it to the list
        pyramid.append(reduced)

    return pyramid


def lapl_pyramid(gPyramid):

    pyramid = []
    levels = len(gPyramid) - 1

    for l in range(levels):

        height = gPyramid[l].shape[0]
        width = gPyramid[l].shape[1]

        expanded = myUpsample(gPyramid[l+1])
        expanded = expanded[:height, :width]

        pyramid.append(gPyramid[l] - expanded)

    pyramid.append(gPyramid[-1])

    return pyramid


def local_blend(gPyramid_mask, lapPyramid_src, lapPyramid_trgt):

    pyramid = []
    levels = len(gPyramid_mask)

    for l in range(levels):

        img_src = lapPyramid_src[l];
        img_trgt = lapPyramid_trgt[l];
        mask = gPyramid_mask[l]

        # Zero Padding
        blended = np.zeros(mask.shape)

        height = mask.shape[0]
        width = mask.shape[1]

        for row in range(height):
            for col in range(width):

                src = mask[row, col] * img_src[row, col]
                trgt = (1 - mask[row, col]) * img_trgt[row, col]

                blended[row, col] = src + trgt

        pyramid.append(blended)

    return pyramid


def collapsePyramid(blended):

    levels = len(blended) -1

    for l in range(levels, 0, -1):

        height = blended[l-1].shape[0]
        width = blended[l-1].shape[1]

        expanded = myUpsample(blended[l])

        # handle cropping
        if expanded.size > blended[l-1].size:
            expanded = expanded[:height, :width]

        blended[l-1] += expanded
    return blended[0]


# Pyramid Blend
def PyramidBlend(source, mask, target):

    # Number of levels in pyramid
    numLevels = 5

    # Build Gaussian Pyramids
    # for source
    gPyramid_src = gauss_pyramid(source, numLevels)
    # for target
    gPyramid_trgt = gauss_pyramid(target, numLevels)
    # for mask
    gPyramid_mask = gauss_pyramid(mask, numLevels)


    # Build Laplacian Pyramids
    # for source
    lapPyramid_src = lapl_pyramid(gPyramid_src)
    # for target
    lapPyramid_trgt = lapl_pyramid(gPyramid_trgt)

    # Get the Local Blend
    blended = local_blend(gPyramid_mask, lapPyramid_src, lapPyramid_trgt)

    # Generate Collapsed Blended Pyramid
    collapsed = collapsePyramid(blended)

    # return source * mask + target * (1 - mask)
    return collapsed


# Poisson Blend
def PoissonBlend(source, mask, target, isMix):

    return source * mask + target * (1 - mask)

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(1, len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])


        ### The main part of the code ###


        # Implement the PyramidBlend function (Task 1)
        pyramidOutput = PyramidBlend(source, mask, target)

        new = (1/(2*2.25)) * pyramidOutput + 0.5

        # Implement the PoissonBlend function (Task 2)
#        poissonOutput = PoissonBlend(source, mask, target, isMix)


        # Writing the result

        now = datetime.now()
        timestamp = datetime.timestamp(now)


        # plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), new)
        #
        plt.imsave("{}pyramid_{}__".format(outputDir, str(index).zfill(2)) + str(timestamp) + ".jpg", new)

#        if not isMix:
#            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
#        else:
#            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
