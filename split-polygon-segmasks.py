import h5py
import numpy as np
from bs4 import BeautifulSoup
import os

# TODO add code to take vector of line segment and shrink along that direction instead

# variables
h5file = './h5s/test.h5' # string directory of where image h5 file is
xmlfile = './xmls/test.xml' # string directory of where xml is
savedir = './newh5s/' # string directory of where to save h5 arrays for polygons
patchsize = 512 # int pixel width/height of nxn patch from whole image

'''
Takes patch image h5 files, original polygon segmentation mask for whole image, and pixel size of patch
Parses polygons and truncates to give only original polygons / parts of polygons that fit into new patch
Parses 2 vertices at a time because it was originally written to include shrinking of lines along their
original direction, but wasn't implemented.
'''
def h5fromxml(imgh5, xml, savedir, patchsize):  
    with open(xml) as xf: #open each individual file
        data = BeautifulSoup(xf, "xml", multi_valued_attributes=None)
        
        data = data.find('Attribute', {'Name':'tumor budding'})
        tumorBudAnnotations = data.find_parent('Annotation')

        soup = tumorBudAnnotations.find_all('Vertices')
      
    with h5py.File(imgh5, "r") as hf:
        if "x" not in hf.keys(): 
            raise TypeError('Not a tumor patch') # used to skip negative h5 files w/o polygons

        polyh5name = os.path.basename(imgh5) # create new h5 with "Polygons" group, containing each polygon's coords
        polyh5name = os.path.splitext(polyh5name)[0]
        polyh5 = h5py.File("./"+savedir+"/"+xml[:-4]+"_"+polyh5name+"_poly.h5", "w")
        poly = polyh5.create_group("Polygons")

        x = hf['x'][:].min()
        y = hf['y'][:].min()

        polynum = 1

        for polygons in soup: # loop over all polygons in xml file
            polygons = polygons.find_all("Vertex")

            newarray = np.empty(0)
            xpast = 0
            ypast = 0

            for vertex in polygons: # loop over all x/y coords in a single polygon
                
                xvertex = float(vertex["X"]) # nth coords
                yvertex = float(vertex["Y"])
                
                if ( (xpast == 0) and (ypast == 0) ): # skip 1st iteration through points
                    continue
                
                if ( ( (xvertex > (x + patchsize) ) or (xvertex < x) ) and ( (xpast > (x + patchsize) ) or (xpast < x) ) ):
                    continue # skip iteration if both n and n-1 points on polygon above patch max x, or if below patch min x

                if ( ( (yvertex >= (y + patchsize) ) or (yvertex <= y) ) and ( (ypast >= (y + patchsize) ) or (ypast <= y) ) ):
                    continue # skip iteration if both n and n-1 points on polygon above patch max y, or if below patch min y
                        
                # change past coords if out of bounds
                if (xpast > (x + patchsize) ): # if (n-1) x above max dimension, set to max dimension
                    xpast = x + patchsize
                    
                if (xpast < x): # if (n-1) x below min dimension, set to min dimension
                    xpast = x
                    
                if (ypast > (y + patchsize) ): # if (n-1) y above max dimension, set to max dimension
                    ypast = y + patchsize
                    
                if (ypast < y): # if (n-1) y below min dimension, set to min dimension
                    ypast = y
                
                # change current coords if out of bounds
                if (xvertex > (x + patchsize) ): # if (n) x above max dimension, set to max dimension
                    xvertex = x + patchsize
                    
                if (xvertex < x): # if (n) x below min dimension, set to min dimension
                    xvertex = x
                    
                if (yvertex > (y + patchsize) ): # if (n) y above max dimension, set to max dimension
                    yvertex = y + patchsize
                    
                if (yvertex < y): # if (n) y below min dimension, set to min dimension
                    yvertex = y
                
                if (newarray.shape == (0,)): # 2nd pass through, creates 1st (n-1) element and 2nd (n) element
                    newarray = np.array((xpast, ypast))
                    currarray = np.array((xvertex, yvertex))
                    newarray = np.expand_dims(newarray, axis=0)
                    newarray = np.concatenate((newarray, np.expand_dims(currarray, axis=0)), axis = 0) 
                else: # all other passes through, concatenates by adding along bottom unless point already exists
                    pastarray = np.array((xpast, ypast))
                    currarray = np.array((xvertex, yvertex))
                    
                    # if last element does not match new n-1 point, add to array
                    if ( (newarray[-1, 0] != xpast) and (newarray[-1, 1] != ypast) ): 
                        newarray = np.concatenate((newarray, np.expand_dims(pastarray, axis=0)), axis = 0)
                        
                    newarray = np.concatenate((newarray, np.expand_dims(currarray, axis=0)), axis = 0) # add n point
                    
                xpast = xvertex #change nth coords to n-1th for next iteration
                ypast = yvertex
                
            if (newarray.shape != (0,)): # if new array is not blank, save corresponding to polygon per patch
                poly.create_dataset("poly_"+str(polynum), data = newarray)
                polynum += 1

        polyh5.close()

def main():
    h5fromxml(h5file, xmlfile, savedir, 512)

if __name__ == "__main__":
    main()
