"""
Assignment 1 - Starter code

@author: rizuj
Name: Rizu Jain
UIN: 430000753

"""


# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path)
    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535

    # Separating the glass image into R, G, and B channels
    b = image[: height, :] / scalingFactor
    g = image[height: 2 * height, :] / scalingFactor
    r = image[2 * height: 3 * height, :] / scalingFactor
    return r, g, b


# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0)
    shifted = np.roll(shifted, shift[1], axis = 1)
    return shifted


# The main part of the code. Implement the FindShift function
def find_shift(im_channel_shift, im_channel_ref, x_ref, y_ref, max_range):
    
    ssd_min = float('inf')
    ssd_min_idx = [0, 0]

    im_channel_shift = circ_shift(im_channel_shift, [x_ref, y_ref])
    crop = im_channel_ref.shape[0] / 4;
    crop = int(crop)

    im_pad = np.pad(im_channel_shift, ((max_range , max_range ), (max_range , max_range )), mode = 'constant')
    num_row, num_col = im_pad.shape

    # The following code implements circshift
    for x_delta in range(-max_range, max_range+1, 1):
        i = x_delta
        x_crop = im_pad[max_range-i:num_row-max_range-i, :]

        for y_delta in range(-max_range, max_range+1, 1):
            j = y_delta
            x_trans = x_crop[:, max_range-j:num_col-max_range-j]

            ssd = np.sum(np.sum((im_channel_ref[crop:-crop, crop:-crop]- x_trans[crop:-crop, crop:-crop]) ** 2))
            if ssd < ssd_min:
                ssd_min_idx = [i, j]
                ssd_min = ssd

    ssd_min_idx[0] += x_ref
    ssd_min_idx[1] += y_ref
    return ssd_min_idx


# Image Pyramid
#   Translation for high resolution images
#   using pyramids
def image_pyramid(im1, im2,get_shift_function):

    downscale = 1/  2 ** 3

    im1_ds = im1
    im2_ds = im2


    im1_ds = cv2.resize(im1, None, fx = downscale, fy = downscale, interpolation = cv2.INTER_AREA)
    im2_ds = cv2.resize(im2, None, fx = downscale, fy = downscale, interpolation = cv2.INTER_AREA)

    # Max range of translation for 8x downscaled image
    shift = get_shift_function(im1_ds, im2_ds, 0, 0, 20)
    print("Scale: 1/ 8", "Shift:", shift)

    for scale in reversed(range(3)):
        im1_ds = im1
        im2_ds = im2
        downscale = 1/  2 ** scale

        im1_ds = cv2.resize(im1, None, fx = downscale, fy = downscale, interpolation = cv2.INTER_AREA)
        im2_ds = cv2.resize(im2, None, fx = downscale, fy = downscale, interpolation = cv2.INTER_AREA)

        move = [ (t*2)+1 for t in shift]
        shift = get_shift_function(im1_ds, im2_ds, move[0], move[1], 4)

        print("Scale: 1/", 2**scale, "Shift:", shift)

    return shift


def edge_find_shift(im_shift, im_ref, x_ref, y_ref, max_range):
    low_threshold = 150
    high_threshold = 150

    im_shift = circ_shift(im_shift, [x_ref, y_ref])
    crop = im_ref.shape[0] / 4;
    crop = int(crop)

    im_shift_new  = im_shift * 255
    img_shift_8 = im_shift_new.astype(np.uint8)
    canny_shift = cv2.Canny(img_shift_8, low_threshold, high_threshold)/255

    im_ref_new = im_ref * 255
    img_ref_8 = im_ref_new.astype(np.uint8)
    canny_ref = cv2.Canny(img_ref_8, low_threshold, high_threshold)/255
    
    ssd_min_idx = [0,0]
    ssd_min = float('inf')

    im_pad = np.pad(canny_shift, ((max_range , max_range ), (max_range , max_range )), mode = 'constant')
    num_row, num_col = im_pad.shape

    # The following code implements circshift
    for x_delta in range(-max_range, max_range+1, 1):
        i = x_delta
        x_crop = im_pad[max_range-i:num_row-max_range-i, :]

        for y_delta in range(-max_range, max_range+1, 1):
            j = y_delta
            x_trans = x_crop[:, max_range-j:num_col-max_range-j]

            ssd = np.sum(np.sum((canny_ref[crop:-crop, crop:-crop]- x_trans[crop:-crop, crop:-crop]) ** 2))
            if ssd < ssd_min:
                ssd_min_idx = [i, j]
                ssd_min = ssd

    ssd_min_idx[0] += x_ref
    ssd_min_idx[1] += y_ref
    return ssd_min_idx


# Automatic Cropping
#   Remove white, black or other color borders. 
#   Detect the borders or the edge between the border and the image.
def auto_crop(img_in):

    img_32 = img_in.astype(np.float32)
    img = cv2.cvtColor(img_32, cv2.COLOR_BGR2GRAY)
    img *= 255
    img_8 = img.astype(np.uint8)
    blurred = cv2.blur(img_8, (3, 3))
    canny = cv2.Canny(blurred, 150, 150)
        
    num_rows, num_cols = canny.shape
    canny_b = canny > 0
    canny_rows = np.sum(canny_b, 1)
    canny_cols = np.sum(canny_b, 0)

    index_rows = np.nonzero(canny_rows > 100)
    index_cols = np.nonzero(canny_cols > 100)
    
    crop_top = np.nonzero(index_rows[0] < num_rows/10)[0]
    
    if crop_top.shape[0] == 0:
        y1 = 0
    else:
        y1 = index_rows[0][np.max(crop_top)]
    
    crop_bot = np.nonzero(index_rows[0] > 9 * num_rows/10)[0]
    
    if  crop_bot.shape[0]  == 0:
        y2 = 0
    else:
        y2 = index_rows[0][np.min(crop_bot)]


    crop_left = np.nonzero(index_cols[0] < num_cols/10)[0]
    
    if  crop_left.shape[0] == 0:
        x1 = 0
    else:
        x1 = index_cols[0][np.max(crop_left)]
    
    crop_right = np.nonzero(index_cols[0] > 9 * num_cols/10)[0]
    
    if  crop_right.shape[0] == 0:
        x2 = 0
    else:
        x2 = index_cols[0][np.min(crop_right )]

    print("Cropped Shape Left Bottom: ", y1, x1, " till y2, x2 Right Bottom: ", y2 , x2)
    
    ## crop the region
    cropped = img_in[y1+1:y2-1, x1+1:x2-1, :]

    return cropped


# Automatic contrasting
#   rescale image intensities such that the darkest pixel is zero.
#   and the brightest pixel is 1 
def auto_contrast(image,percent_clip):
    
    fraction = 100/percent_clip

    values = np.mean(image,(0,1))
#    print("Before Auto Contrast:", values)
    
    dark_chnl = np.argmax(values)
    bright_chnl = np.argmin(values)
    
    # we wil have to perform percentage stretching as we have pixels in the complete range
    # but we still have tails in our intensity histogram
    img_1d = image[:,:,bright_chnl].flatten()
    
    # ignore all zero valued pixels    
    img_1d = img_1d[np.nonzero(img_1d)]
    
    # take out 2% on both the side
    k = int(len(img_1d)/fraction)
    
    min_idx = np.argpartition(img_1d,k)[:k]
    min_val = np.sort(img_1d[min_idx])[-1]

    img_1d = image[:,:,dark_chnl].flatten()
    
    # ignore all one valued pixels    
    img_1d = img_1d[np.nonzero(img_1d  < 1)]
    
    # take out some percent of intensities on both the side
    k = int(len(img_1d)/fraction)

    max_idx = np.argpartition(-img_1d,k)[:k]
    max_val = np.sort(img_1d[max_idx])[0]
    
    image = (image - min_val) / (max_val  - min_val)
    
    image = np.clip(image,0,1)
    values = np.mean(image,(0,1))
    
#    print("After Auto Contrast:", values)
    return image
    

if __name__ == '__main__':
    
    # Setting the input output file path
    imageDir = '../Images/'
    outDir = '../Results/'
    imageNames = ['monastery.jpg', 'cathedral.jpg', 'nativity.jpg', 'settlers.jpg', 'icon.tif', 'emir.tif', 'harvesters.tif', 'lady.tif', 'self_portrait.tif', 'three_generations.tif', 'train.tif', 'turkmen.tif', 'village.tif']
              
    for imageName in imageNames:
        print("Image : ", imageName)
        start = time.time()
        
        # Get r, g, b channels from image strip
        r, g, b = read_strip(imageDir + imageName)

        # Calculate shift
        x_ref = 0
        y_ref = 0
        max_range = 20

        # For high resolution images use an image pyramid for translation
        if r.shape[0] > 3076:
            print(" -- - Multi Scale Approach -- -")
            print("R channel:: ")
            rShift = image_pyramid(r, b,find_shift)
            print("G channel:: ")
            gShift = image_pyramid(g, b,find_shift)

        else:
            print(" -- - Single Scale Approach -- -")
            rShift = find_shift(r, b, x_ref, y_ref, max_range)
            print("R channel::    (x, y) Translation: ", rShift)
            gShift = find_shift(g, b, x_ref, y_ref, max_range)
            print("G channel::    (x, y) Translation: ", gShift)

        # Shifting the images using the obtained shift values
        finalB = b
        finalG = circ_shift(g, gShift)
        finalR = circ_shift(r, rShift)
        
        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)
        
        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] + '.jpg', finalImage)

        end = time.time()
        print("Time : ", int(end - start), "sec")
        print("Done")
        print(" == == == == == == == == == == ")


# Better Features
        start = time.time()
        r, g, b = read_strip(imageDir + imageName)
        
        print("Better Features: Image Alignment using Edges ...")
        
        if r.shape[0] > 3076:
            print(" -- - Multi Scale Approach -- -")
            print("R channel:: ")
            rShift = image_pyramid(r, b,edge_find_shift)
            print("G channel:: ")
            gShift = image_pyramid(g, b,edge_find_shift)

        else:
            print(" -- - Single Scale Approach -- -")
            rShift = edge_find_shift(r, b, x_ref, y_ref, max_range)
            print("R channel::    (x, y) Translation: ", rShift)
            gShift = edge_find_shift(g, b, x_ref, y_ref, max_range)
            print("G channel::    (x, y) Translation: ", gShift)

        # Shifting the images using the obtained shift values
        finalB = b
        finalG = circ_shift(g, gShift)
        finalR = circ_shift(r, rShift)

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)
        
        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] +  '_better_features' + '.jpg', finalImage)

        end = time.time()
        print("Time : ", int(end - start), "sec")
        print("Done")


# Auto cropping 
        print("Auto Cropping ... ")
        cropped = auto_crop(finalImage)

        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] + '_crop' +'.jpg', cropped)
        print("Done")


# Auto Contrasting
        print("Auto Contrasting ...")
        contrasted = auto_contrast(cropped, percent_clip=4)

        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] + '_contrast' + '.jpg', contrasted)
        print("Done")
                
        print(" == == == == == == == == == == ")
        
