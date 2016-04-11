#########################################################################
# Patch Extractor:
#   Takes as input a single image and returns a 
#   list of patches (each resized to 1-D) using
#   PatchSize and PatchStride.
#   Note: Only Luminance (Y) patches are returned.
#########################################################################
from dependency import *

class patch_generator:
    def __init__(self,imagefolder,psize,stride,CenterSize):
        self.patches,self.u_vals,self.v_vals = self.PatchExtractor(imagefolder,psize,stride,CenterSize)

    def PatchExtractor(a,image,psize,stride=None):
        # print image
        # print psize
        # print stride
        assert(psize%2 == 1) #--- Patch size must be odd

        patches = []
        if stride is None:
            stride = psize
        utrain = []
        vtrain = []
        

        dim = 0 #--- Use only Y (Luminance)
        # print 'hell'
        #--- Iterates over all possible patch centers in the image which give a full patch
        for ny in range( (1+image.shape[1]-psize)//stride ):
            for nx in range( (1+image.shape[0]-psize)//stride ):
                #--- Get patch for current center pixel:
                patch = image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim]
                patch = patch - np.average(patch)
                patches.append( patch.reshape((1,-1)))
                
                #--- Get UV values for current center pixel:
                ut = image[(nx*stride)+(psize//2) - CenterSize:(nx*stride)+(psize//2) + 1 + CenterSize,(ny*stride)+(psize//2) - CenterSize:(ny*stride)+(psize//2) + 1 + CenterSize,1]
                ut = image[(nx*stride)+(psize//2) - CenterSize:(nx*stride)+(psize//2) + 1 + CenterSize,(ny*stride)+(psize//2) - CenterSize:(ny*stride)+(psize//2) + 1 + CenterSize,2]
                utrain.append( ut.reshape((1,-1)) )
                vtrain.append( ut.reshape((1,-1)) )
                # utrain[nx,ny] = image[(nx*stride)+(psize//2),(ny*stride)+(psize//2),1]
                # vtrain[nx,ny] = image[(nx*stride)+(psize//2),(ny*stride)+(psize//2),2]

        #--- UV Values to train for regression
        # utrain = utrain.reshape((-1,1))
        # vtrain = vtrain.reshape((-1,1))
        return np.vstack(patches),np.vstack(utrain),np.vstack(vtrain)
