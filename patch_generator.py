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
        self.patches,self.patch_means,self.u_vals,self.v_vals = self.PatchExtractor(imagefolder,psize,stride,CenterSize)

    def PatchExtractor(self,imagefolder,psize,stride=None,CenterSize=1):
        assert(psize%2 == 1) #--- Patch size must be odd

        patches = []
        patch_means = []
        if stride is None:
            stride = psize
        utrain = []
        vtrain = []
        
        dim = 0 #--- Use only Y (Luminance)
        # print 'hell'
        #--- Iterates over all possible patch centers in the image which give a full patch
        for file in listdir(imagefolder):
            B = cv2.imread(imagefolder+'/'+file)
            image = cv2.cvtColor(B,cv2.COLOR_BGR2YUV)
            for ny in range( (1+image.shape[1]-psize)//stride ):
                for nx in range( (1+image.shape[0]-psize)//stride ):
                    #--- Get patch for current center pixel:
                    patch = image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim]
                    patch_means.append(np.average(patch))
                    patch = patch - np.average(patch)
                    patches.append( patch.reshape((1,-1)))
                    
                    #--- Get UV values for current center pixel:
                    ut = image[(nx*stride)+(psize//2) - CenterSize:(nx*stride)+(psize//2) + 1 + CenterSize,(ny*stride)+(psize//2) - CenterSize:(ny*stride)+(psize//2) + 1 + CenterSize,1]
                    vt = image[(nx*stride)+(psize//2) - CenterSize:(nx*stride)+(psize//2) + 1 + CenterSize,(ny*stride)+(psize//2) - CenterSize:(ny*stride)+(psize//2) + 1 + CenterSize,2]
                    ut = ut - np.average(patch)
                    vt = vt - np.average(patch)
                    utrain.append( ut.reshape((1,-1)) )
                    vtrain.append( vt.reshape((1,-1)) )


        ############################
        #will have to pass the patch mean with each patch

        #--- UV Values to train for regression
        # utrain = utrain.reshape((-1,1))
        # vtrain = vtrain.reshape((-1,1))
        return np.vstack(patches),patch_means,np.vstack(utrain),np.vstack(vtrain)
