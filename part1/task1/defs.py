import numpy as np

def simple_image_write(image):

    ''' 
    Function that writes given image file into a binary file.
    
    Parameters
    ----------
    image : array-like, shape(image.shape)
        Matrix of voxel/pixel values representing given image.
        
    Returns
    -------
    outfile : binary file
        Image values stored in binary format.
    '''

    outfile = '../../data/image.sim'
    image.tofile(outfile)

    return outfile

def simple_image_read(outfile):

    ''' 
    Function that reads the previously written binary file so it can be viewed.
    
    Parameters
    ----------
    outfile : binary file
        Contains image values in binary format.
        
    Returns
    -------
    output : image file
        Matrix of pixel/voxel values representing the image previously stored in outfile.
    
    '''

    output = np.fromfile(outfile, dtype='int16')

    return output
