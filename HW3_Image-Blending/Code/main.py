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
#
    source = plt.imread(path + "source_" + id + ".jpg") / 255
    target  = plt.imread(path + "target_" + id + ".jpg") / 255
    mask = plt.imread(path + "mask_" + id + ".jpg") / 255

    return source, mask, target


# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset

    if (xOffset < 0):
        mask = mask[abs(xOffset):, :]
        source = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask = mask[:, abs(yOffset):]
        source = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask = mask[:sourceHeight, :]
        source = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask = mask[:, :sourceWidth]
        source = source[:, :sourceWidth]

    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight,
        yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight,
        yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal


def myUpsample(img_in):
    img_out = cv2.pyrUp(img_in)
    return img_out


def myDownsample(img_in):
    img_out = cv2.pyrDown(img_in)
    return img_out


# Construction of gaussian pyramid
def gaussianPyramid(img_in, numLevels):

    # a list that will contain all the samples of the img_in.
    pyramid = [img_in]

    for l in range(numLevels):

        # Downsample the last level of image
        reduced = myDownsample(pyramid[-1])

        # Append it to the list
        pyramid.append(reduced)

    return pyramid

# Construction of laplacian pyramid from gaussian pyramid


def laplacianPyramid(gPyramid):

    # a list that will contain all the samples of the laplacian tranform.
    pyramid = []
    levels = len(gPyramid) - 1

    for l in range(levels):

        height = gPyramid[l].shape[0]
        width = gPyramid[l].shape[1]

        # upsample the gaussian layer
        expanded = myUpsample(gPyramid[l+1])

        # if the size of the expanded image gets larger
        # crop the exanded image with shape of the relative layer.
        expanded = expanded[:height, :width]

        # Subtract it from the previous layer
        pyramid.append(gPyramid[l] - expanded)

    # append the last one
    pyramid.append(gPyramid[-1])

    return pyramid


# Feathering of image layers from pyramids of source and target
# As per slides from the lecture
# Iblend = αIleft + (1-α)Iright
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

                # if mask = 1 (white region), take from source
                # if mask = 0 (black region), take from target
                src = mask[row, col] * img_src[row, col]
                trgt = (1 - mask[row, col]) * img_trgt[row, col]

                # Note: check dimension.
                # should be same as that of src & trgt
                blended[row, col] = src + trgt

        pyramid.append(blended)

    return pyramid


# take the input pyramid >> collapse it
def collapsePyramid(blended):

    levels = len(blended) - 1

    # start from the smallest layer
    # keep on adding to next layer
    # until the largest layer
    for l in range(levels, 0, -1):

        height = blended[l-1].shape[0]
        width = blended[l-1].shape[1]

        # expand smaller layer to be added with larger layer
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

    return collapsed


# Poisson Blend 
def PoissonBlend(source_orig, mask, target_orig, isMix):

    source = source_orig * 255
    source = source.astype(np.uint8)
    
    target = target_orig * 255
    target = target.astype(np.uint8)
    
    width_y = target.shape[0]
    height_x = target.shape[1]

    # Generate the Poisson matrix A
    
    # Generate a square empty sparce matrix of size height_x X height_x
    # row wise
    filter = scipy.sparse.lil_matrix((height_x, height_x))
    
    # Set diagonal elements of the laplacian filter 
    # center element >> -4 , right, left, up , down >> 1
    filter.setdiag(-1, -1)
    filter.setdiag(4)
    filter.setdiag(-1, 1)

    # Build a block diagonal sparse matrix from provided matrix filter
    # list of lists format is used for this diagonal sparse matrix
    matrixA = scipy.sparse.block_diag([filter] * width_y).tolil()
    matrixA.setdiag(-1, 1*height_x)
    matrixA.setdiag(-1, -1*height_x) 
    
    # store the column compressed format to generate matrix B later.
    temp_matrixA = matrixA.tocsc()  

    # iterate through each pixel of the mat_A
    # check its position w.r.t. matrixA
    for y in range(1, width_y - 1):
        for x in range(1, height_x - 1):
            
            # if the mask region is black, we need color from target
            # set the values outside the mask to 1
            # else 0
            if mask[y, x] == 0:
                
                pixel_pos = x + y * height_x
                matrixA[pixel_pos, pixel_pos - height_x] = 0
                matrixA[pixel_pos, pixel_pos - 1] = 0
                matrixA[pixel_pos, pixel_pos] = 1
                matrixA[pixel_pos, pixel_pos + 1] = 0
                matrixA[pixel_pos, pixel_pos + height_x] = 0
                

    # Convert to compressed column format
    matrixA = matrixA.tocsc()

    mask_flat = mask.flatten()
    for channel in range(3):
        source_flat = source[0:width_y, 0:height_x, channel].flatten()
        target_flat = target[0:width_y, 0:height_x, channel].flatten()


        # Generate the B matrix of the equation
        
        # the following line of code sets the source values
        # for the pixel inside the mask (white region)      

        matrixB =  temp_matrixA.dot(source_flat)
        
        if isMix == True:
            matrixB_trgt =  temp_matrixA.dot(target_flat)
            matrixB = (matrixB +matrixB_trgt )/2

        matrixB[mask_flat == 0] = target_flat[mask_flat == 0]

        matrixX = spsolve(matrixA, matrixB)
        matrixX = matrixX.reshape((width_y, height_x))
        matrixX[matrixX > 255] = 255
        matrixX[matrixX < 0] = 0
        matrixX = matrixX.astype('uint8')

        target[0:width_y, 0:height_x, channel] = matrixX


    return target


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [
        140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88], [-16,386]] 


    # main area to specify files and display blended image
    for index in range(1, len(offsets)):
        
        # Read data and clean mask
        # source, maskOriginal, target = Read(str(index).zfill(2), inputDir)
        source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0    


        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])


        ### The main part of the code ###

        # Implement the PyramidBlend function (Task 1)
        pyramidOutput = PyramidBlend(source, mask, target)
        pyramidOutput  = np.clip(pyramidOutput,0,1)


        # Implement the PoissonBlend function (Task 2)'
        mask_1D = mask[:,:,0]
        poissonOutput = PoissonBlend(source, mask_1D, target, isMix)
        

        ### Writing the result
        now = datetime.now()
        timestamp = datetime.timestamp(now)

        plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
        
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
        else:
            cv2.imwrite("{}poisson_{}_Mixing.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
