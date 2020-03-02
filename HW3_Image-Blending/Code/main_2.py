# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:40:06 2020

@author: rizuj
"""

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

import scipy.sparse
from scipy.sparse.linalg import spsolve


# Read source, target and mask for a given id
def Read(id, path=""):
    source = cv2.imread(path + "source_" + id + ".jpg") / 255
    target = cv2.imread(path + "target_" + id + ".jpg") / 255
    mask = cv2.imread(path + "mask_" + id + ".jpg") / 255

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


def gaussianPyramid(img_in, numLevels):

    # Create a list that will contain all the samples of the img_in.
    pyramid = [img_in]

    for l in range(numLevels):
        # Downsample the last level of image
        reduced = myDownsample(pyramid[-1])
        # Append it to the list
        pyramid.append(reduced)

    return pyramid


def laplacianPyramid(gPyramid):

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

        img_src = lapPyramid_src[l]
        img_trgt = lapPyramid_trgt[l]
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

    levels = len(blended) - 1

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
    gPyramid_src = gaussianPyramid(source, numLevels)
    # for target
    gPyramid_trgt = gaussianPyramid(target, numLevels)
    # for mask
    gPyramid_mask = gaussianPyramid(mask, numLevels)

    # Build Laplacian Pyramids
    # for source
    lapPyramid_src = laplacianPyramid(gPyramid_src)
    # for target
    lapPyramid_trgt = laplacianPyramid(gPyramid_trgt)

    # Get the Local Blend
    blended = local_blend(gPyramid_mask, lapPyramid_src, lapPyramid_trgt)

    # Generate Collapsed Blended Pyramid
    collapsed = collapsePyramid(blended)

    # return source * mask + target * (1 - mask)
    return collapsed


def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask):

    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()
    
    m_x_range= mask.shape[0] 
    m_y_range = mask.shape[1]

    # set the region outside the mask to identity    
    for y in range(1, m_x_range - 1):
        for x in range(1,m_y_range  - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x
    
    
    return target

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
    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [
        140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

#    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [
#        140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(1, len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0
        print("source " , source.shape)
        print("mask ",mask.shape)
        print("target ",target.shape)
        print("\n")
    

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])
        
        print("source " , source.shape)
        print("mask ",mask.shape)
        print("target ",target.shape)
        print("\n")
    
        
#        cv2.imwrite("{}source_tmp_{}.jpg".format(outputDir, str(index).zfill(2)), source)


        ### The main part of the code ###

        # Implement the PyramidBlend function (Task 1)
        pyramidOutput = PyramidBlend(source, mask, target)
        pyramidOutput  = np.clip(pyramidOutput,0,1)
        
        # Implement the PoissonBlend function (Task 2)
        # poissonOutput = PoissonBlend(source, mask, target, isMix)



        #pOut = poisson_edit(source, target, mask[:,:,0])
        
#        poissonOutput = source * mask + target * (1 - mask)

        # Writing the result

        now = datetime.now()
        timestamp = datetime.timestamp(now)

        plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
#        plt.imsave("{}pyramid_{}__".format(outputDir, str(
#            index).zfill(2)) + str(timestamp) + ".jpg", pyramidOutput)
        
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index).zfill(2)), pOut)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index).zfill(2)), pOut)
