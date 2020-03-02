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
    source = plt.imread(path + "source_" + id + ".jpg") / 255
    target = plt.imread(path + "target_" + id + ".jpg") / 255
    mask = plt.imread(path + "mask_" + id + ".jpg") / 255

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
    """
    Generate the Poisson matrix.
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)

    return mat_A


# Poisson Blend
def PoissonBlend(source, mask, target, isMix, offset):
    
    width_y = target.shape[0]
    height_x = target.shape[1]      
    
    plt.imsave("../Results/source_temp_07.jpg", source)
    plt.imsave("../Results/mask_temp_07.jpg", mask)
    plt.imsave("../Results/target_temp_07.jpg", target)    
    
    mat_A = laplacian_matrix(width_y, height_x)
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity
    for y in range(1, width_y - 1):
        for x in range(1, height_x  - 1):
            if mask[y, x] == 0:
                k = x + y * height_x
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + height_x] = 0
                mat_A[k, k - height_x] = 0

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(3):
        source_flat = source[0:width_y,
                             0:height_x, channel].flatten()
        target_flat = target[0:width_y, 0:height_x, channel].flatten()

        # inside the mask:
        mat_b = laplacian.dot(source_flat)

        # outside the mask:
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        print(x.shape)
        x = x.reshape((width_y, height_x))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        target[0:width_y, 0:height_x, channel] = x
        print(x.shape)

    print(target.shape)
    cv2.imwrite("figs/out/target_temp_07_end.jpg", target)

    return target
    # return source * mask + target * (1 - mask)


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
#    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [
#        140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

#    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [
#        140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
#    for index in range(1, len(offsets)):
#        # Read data and clean mask
#        #source, maskOriginal, target = Read(str(index).zfill(2), inputDir)
#        source, maskOriginal, target = Read(str(index).zfill(2), inputDir)
#
#        # Cleaning up the mask
#        mask = np.ones_like(maskOriginal)
#        mask[maskOriginal < 0.5] = 0    

#    source = plt.imread("../Images/source_07.jpg") / 255
#    target = plt.imread("../Images/target_07.jpg") / 255
#    maskOriginal = plt.imread("../Images/mask_07.jpg") / 255

    source = cv2.imread("../Images/source_07.jpg") / 255
    target = cv2.imread("../Images/target_07.jpg") / 255
    mask = cv2.imread("../Images/mask_07.jpg") / 255


#    mask = np.ones_like(maskOriginal)
#    mask[maskOriginal < 0.5] = 0    


    # Align the source and mask using the provided offest
    source, mask = AlignImages(mask, source, target, [20, 20])

        
        ### The main part of the code ###

        # Implement the PyramidBlend function (Task 1)
#        pyramidOutput = PyramidBlend(source, mask, target)
#        pyramidOutput  = np.clip(pyramidOutput,0,1)
        
#         # Implement the PoissonBlend function (Task 2)'
    mask_1D = mask[:,:,0]
    poissonOutput = PoissonBlend(source, mask_1D, target, isMix, [20,20])
    
    cv2.imwrite("../Results/poisson_out_07.jpg", poissonOutput)
    
    #mat_A ,laplacian = PoissonBlend(source, mask[:,:,0], target, isMix)
    
    # plt.imsave("../Results/poisson_out_07.jpg", poissonOutput)


#         # Writing the result
#         now = datetime.now()
#         timestamp = datetime.timestamp(now)

#         plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
# #        plt.imsave("{}pyramid_{}__".format(outputDir, str(
# #            index).zfill(2)) + str(timestamp) + ".jpg", pyramidOutput)
        
#         if not isMix:
#             plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index).zfill(2)), pOut)
#         else:
#             plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index).zfill(2)), pOut)
