from dependency import *

class AssignColor:
    def __init__(self,image,kmodel,u_reg_models,v_reg_models,psize,CenterSize):
        self.output_yuv = self.assignColor(image,kmodel,u_reg_models,v_reg_models,psize,CenterSize)


    def assignColor(self,image,kmodel,u_reg_models,v_reg_models,psize,CenterSize): #Image should be YUV
        CenterLen = (2*CenterSize + 1)*(2*CenterSize + 1)
        p_by2 = psize//2
        patches = []
        patch_centers = []
        patch_means =[]
        for px in range( p_by2, image.shape[0]-p_by2 ):
            for py in range( p_by2, image.shape[1]-p_by2 ):
                patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
                patch_means.append(np.average(patch))
               # patch = patch - np.average(patch)
                patches.append( patch.reshape((1,-1)) )
                patch_centers.append( [px,py] )

        patches = np.vstack(patches)
        mean_labels = kmodel.predict( patches )

        
        for ind in range(len(patch_centers)):#iterate over all patches(in a way all patches of this cluster centre)
            
            # u_patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
            # v_patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
            i = mean_labels[ind]
            #u_vals = np.zeros(CenterLen)
            #v_vals = np.zeros(CenterLen)

            # for j in range(CenterLen):
            #     u_vals[j] = (1.0/CenterLen)*( u_reg_models[i][j].predict( patches[ind].reshape(1,-1)))# + patch_means[ind])#stores u values for all pixels predicted by this patch
            #     v_vals[j] = (1.0/CenterLen)*( v_reg_models[i][j].predict( patches[ind].reshape(1,-1)))# + patch_means[ind])

            u_val = u_reg_models[i][0].predict( patches[ind].reshape(1,-1) )
            v_val = v_reg_models[i][0].predict( patches[ind].reshape(1,-1) )
            
            px,py = patch_centers[ind][0], patch_centers[ind][1]
            image[px,py,1] = u_val
            image[px,py,2] = v_val
            
            #--- Progress reporting section
            if ind%1000 == 0:
                string = "\rColor Assign Progress: %5.2f%%" % ( 100.0*ind/len(patch_centers))
                sys.stdout.write(string)
                sys.stdout.flush()
        img = image
        uchannel = img[:,:,1]
        uchannel = cv2.GaussianBlur(uchannel, (5,5),0)
        vchannel = img[:,:,2]
        vchannel = cv2.GaussianBlur(vchannel, (5,5),0)
        img[:,:,1] = uchannel
        img[:,:,2] = vchannel
        
        return np.uint8(img)
