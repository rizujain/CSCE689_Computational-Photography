""" 
Assignment 6 - Starter code

"""

import numpy as np
import os
import math

import matplotlib.pyplot as plt
from gsolve import gsolve

# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Results/'

_lambda = 50

calibSetName = 'Office'
file_prefix = '../Results/' + calibSetName + '_'

# Parsing the input images to get the file names and corresponding exposure
# values
filePaths, exposures = ParseFiles(calibSetName, inputDir)


""" Task 1 """

# Sample the images

images = []

for file in filePaths:
    image = plt.imread(file)
    if calibSetName == 'Chapel':
        image = image * 255.99
    image = image.astype(np.uint8)
    images.append(image)    
    

B = np.zeros((len(exposures)))
for l in range(len(exposures)):
    B[l] = math.log2(exposures[l])


sample_size = int(5 * 256 / (len(images)-1))
indices = np.random.randint(0,images[0].shape[1] * images[0].shape[0],sample_size)


pixel_samples_r = np.zeros((len(images),sample_size),np.uint8)
pixel_samples_g = np.zeros((len(images),sample_size),np.uint8)
pixel_samples_b = np.zeros((len(images),sample_size),np.uint8)

for idx,image in enumerate(images):
    channel = image[:,:,0].flatten()
    pixel_samples_r[idx,:]=channel[indices]
    channel = image[:,:,1].flatten()
    pixel_samples_g[idx,:]=channel[indices]
    channel = image[:,:,2].flatten()
    pixel_samples_b[idx,:]=channel[indices]

# Create the triangle function
zees = list(range(256))

zmax = max(zees)
zmin = min(zees)
zavg = (int(zmax)+zmin)/2

w = np.zeros((zmax+1))

for z in zees:
    if z > zavg:
        w[z] = zmax-z
    else:
        w[z] = z -zmin

    
# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
g_r,_ =  gsolve(pixel_samples_r.T,B,_lambda,w)
g_g,_ =  gsolve(pixel_samples_g.T,B,_lambda,w)
g_b,_ =  gsolve(pixel_samples_b.T,B,_lambda,w)

plt.plot(g_r,zees)
plt.plot(g_g,zees)
plt.plot(g_b,zees)
plt.savefig(file_prefix+"CRF.png")

""" Task 2 """

def rad(x,chnl):
    if chnl == 'r':
        return g_r[x]
    if chnl == 'g':
        return g_g[x]
    if chnl == 'b':
        return g_b[x]
    
rad_vec = np.vectorize(rad)

# generate irradiance

denom = np.zeros((image.shape))
img_irrad= np.zeros((image.shape))

# Reconstruct the radiance using the calculated CRF
for idx,image in enumerate(images):
    new_image = np.zeros((image.shape))
    new_image[:,:,0] = rad_vec(image[:,:,0],'r')
    new_image[:,:,1] = rad_vec(image[:,:,1],'g')
    new_image[:,:,2] = rad_vec(image[:,:,2],'b')
    new_image = new_image - B[idx]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for ch in range(image.shape[2]):
                new_image[i,j,ch] = w[image[i,j,ch]] * new_image[i,j,ch]
                denom[i,j,ch] += w[image[i,j,ch]]

    img_irrad += new_image   


# final irradiance image
denom[denom==0] = 1e-10
lnei = img_irrad / denom
ei = np.exp(lnei)

#img_irrad = np.exp(img_irrad)
#
gamma = 0.1
#
img_global_tone = ei/np.max(ei) 
img_global_tone = np.power(img_global_tone,gamma)
img_global_tone = np.clip(img_global_tone,0,1)

img_global_tone  = img_global_tone  * 255.99

img_global_tone  = img_global_tone.astype(np.uint8)

plt.imsave(file_prefix+"global.png",img_global_tone)    

""" Task 3 """

I = np.zeros((ei.shape[0],ei.shape[1]))

for i in range(ei.shape[2]):
    I += ei[:,:,i]

I = I /3

chr_chnls = np.zeros((ei.shape))

for i in range(ei.shape[2]):
    chr_chnls[:,:,i] = ei[:,:,i] / I


L = np.log2(I)

from scipy.ndimage.filters import gaussian_filter

blurred = gaussian_filter(L, sigma=0.5)
D = L - blurred

dR = 4
s = dR/(np.max(blurred)-np.min(blurred))
b_dash = (blurred - np.max(blurred)) * s

b_dash_D = b_dash + D

def raise2(x):
    return  2 ** x

raise2_vec = np.vectorize(raise2)

O = raise2_vec(b_dash_D)

for i in range(ei.shape[2]):
    chr_chnls[:,:,i] = chr_chnls[:,:,i] * O

img_local_tone = np.power(chr_chnls,0.6)
img_local_tone = np.clip(img_local_tone ,0,1)

img_local_tone   = img_local_tone * 255.99

img_local_tone = img_local_tone.astype(np.uint8)

plt.imsave(file_prefix+"local.png",img_local_tone)    

# Perform both local and global tone-mapping

    

