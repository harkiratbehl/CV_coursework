from dependency import *

class AssignColor:
    def __init__(self,image,kmodel,u_reg_models,v_reg_models,psize):
        self.output_yuv = self.assignColor(image,kmodel,u_reg_models,v_reg_models,psize)


    def assignColor(a,image,kmodel,u_reg_models,v_reg_models,psize): #Image should be YUV
        p_by2 = psize//2
        patches = []
        patch_centers = []
        for px in range( p_by2, image.shape[0]-p_by2 ):
            for py in range( p_by2, image.shape[1]-p_by2 ):
                patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
                patches.append( patch.reshape((1,-1)) )
                patch_centers.append( [px,py] )

        patches = np.vstack(patches)
        mean_labels = kmodel.predict( patches )

        for i,pt in enumerate(kmodel.cluster_centers_):
            #--- The indices for the patches which belong to current cluster
            indices = [mean_labels == i]
            for ind in range(len(patch_centers)):
                if indices[0][ind] is False:
                    continue
                # u_patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
                # v_patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
                u_val = u_reg_models[i].predict( patches[ind].reshape(1,-1) )
                v_val = v_reg_models[i].predict( patches[ind].reshape(1,-1) )
                
                px,py = patch_centers[ind][0], patch_centers[ind][1]
                image[px,py,1] = u_val
                image[px,py,2] = v_val
                
                #--- Progress reporting section
                if ind%10000 == 0:
                    string = "\rProgress: %5.2f " % ( 100.0*ind/len(patch_centers) )
                    string += " Last u,v assigned: %3.2f %3.2f" % (u_val,v_val)
                    sys.stdout.write(string)
                    sys.stdout.flush()
            print "\nPhase completed: ", i+1," out of ",len(kmodel.cluster_centers_)
        
        return np.uint8(image)