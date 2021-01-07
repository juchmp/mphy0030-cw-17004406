import numpy as np
from array import array

def simple_image_write(image):

    ''' Function that writes given image file into a binary file '''

    outfile = '../../data/image.sim'
    image.tofile(outfile)

    return outfile

def simple_image_read(outfile):

    ''' Function that reads the previously written binary file so it can be viewed '''

    #with open('../../data/image.sim', 'r') as bfile:
    output = np.fromfile(outfile, dtype='int16')

    return output
