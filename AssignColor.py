from dependency import *

class AssignColor:
    def __init__(self,image,kmodel,u_reg_models,v_reg_models,psize,CenterSize):
        self.output_yuv = self.assignColor(image,kmodel,u_reg_models,v_reg_models,psize,CenterSize)


    def assignColor(self,image,kmodel,u_reg_models,v_reg_models,psize,CenterSize): #Image should be YUV
        CenterLen = (2*CenterSize + 1)*(2*CenterSize + 1)
        p_by2 = psize//2
        print "Assigning Color"
        patches = []
        patch_centers = []
        patch_means =[]
        ind = 0
        prog = ((image.shape[0]-psize)*(image.shape[1] - psize))/(psize*psize)
        for px in range( p_by2, image.shape[0]-p_by2 ):
            for py in range( p_by2, image.shape[1]-p_by2 ):
                patch = image[ px-p_by2:1+px+p_by2, py-p_by2:1+py+p_by2, 0]
               
               # patch_means.append(np.average(patch))
               # patch = patch - np.average(patch)
                #patches.append( patch.reshape((1,-1)) )
                #patch_centers.append( [px,py] )
                patch = patch.reshape((1,-1))
                patch = np.float64(patch)
                i = kmodel.predict(patch.reshape(1,-1))

                u_val = u_reg_models[i][0].predict( patch.reshape(1,-1) )
                v_val = v_reg_models[i][0].predict( patch.reshape(1,-1) )
            
                image[px,py,1] = u_val
                image[px,py,2] = v_val
                
            #--- Progress reporting section
                if ind%1000 == 0:
                    string = "\rColor Assign Progress: %5.2f%%" % ( 100.0*ind/prog)
                    sys.stdout.write(string)
                    sys.stdout.flush()                
                ind = ind + 1
        
        img = image
        uchannel = img[:,:,1]
        uchannel = cv2.GaussianBlur(uchannel, (5,5),0)
        vchannel = img[:,:,2]
        vchannel = cv2.GaussianBlur(vchannel, (5,5),0)
        img[:,:,1] = uchannel
        img[:,:,2] = vchannel
        
        return np.uint8(img)
