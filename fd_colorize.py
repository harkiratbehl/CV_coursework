#!/usr/bin/python

import cv2
import os,sys,copy,pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

#--- Global declarations
NumClusters = 20
PatchSize = 15
PatchStride = PatchSize

#########################################################################
# Patch Extractor:
#   Takes as input a single image and returns a 
#   list of patches (each resized to 1-D) using
#   PatchSize and PatchStride.
#   Note: Only Luminance (Y) patches are returned.
#########################################################################

def PatchExtractor(image,psize,stride=None):
    assert(psize%2 == 1) #--- Patch size must be odd

    patches = []
    utrain = np.zeros(( (1+image.shape[0]-psize)//stride, (1+image.shape[1]-psize)//stride ))
    vtrain = np.zeros(( (1+image.shape[0]-psize)//stride, (1+image.shape[1]-psize)//stride ))
    if stride is None:
        stride = psize

    dim = 0 #--- Use only Y (Luminance)
    
    #--- Iterates over all possible patch centers in the image which give a full patch
    for ny in range( (1+image.shape[1]-psize)//stride ):
        for nx in range( (1+image.shape[0]-psize)//stride ):
            #--- Get patch for current center pixel:
            patch = image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim]
            patches.append( patch.reshape((1,-1)))
            
            #--- Get UV values for current center pixel:
            utrain[nx,ny] = image[(nx*stride)+(psize//2),(ny*stride)+(psize//2),1]
            vtrain[nx,ny] = image[(nx*stride)+(psize//2),(ny*stride)+(psize//2),2]

    #--- UV Values to train for regression
    utrain = utrain.reshape((-1,1))
    vtrain = vtrain.reshape((-1,1))
    return np.vstack(patches),utrain,vtrain

#########################################################################
# TrainKmeansAndRegression:
#   Trains the K-means clutering model and uses
#   the U-V values of the patch centers to train
#   regression models for each cluster center
#########################################################################

def TrainKmeansAndRegression(patches,u_vals,v_vals):
    # Model declarations
    lrmodel = LinearRegression(n_jobs=-1)
    kmodel = KMeans(n_clusters=NumClusters,n_jobs=-1)

    labels = kmodel.fit_predict(patches)
    print "KMeans clustering completed."

    u_reg_models = []
    v_reg_models = []

    for i,pt in enumerate(kmodel.cluster_centers_):
        nearest_points = patches[labels == i,]
        u_regress = u_vals[labels == i]
        v_regress = v_vals[labels == i]

        #--- To verify that average of the spliced points is actually the mean_patch, use this line:
        #--- print "Average: ",np.average(nearest_points,axis=0),"\nvs mean:",pt
        # print "nearest_points:",nearest_points.shape

        lrmodel.fit(nearest_points,u_regress)
        u_reg_models.append( copy.deepcopy(lrmodel) )
        lrmodel.fit(nearest_points,v_regress)
        v_reg_models.append( copy.deepcopy(lrmodel) )

        string = "\rTraining regression models... %4.2f " % ( 100.0*i/len(kmodel.cluster_centers_) )
        sys.stdout.write(string)
        sys.stdout.flush()
    return kmodel,u_reg_models,v_reg_models

#########################################################################
# AssignColor:
#   Takes as input the target gray-scale image, the 
#   trained k-means model and UV regression model lists,
#   and the patch size.
#   Returns final colorized image.
#########################################################################

def AssignColor(image,kmodel,u_reg_models,v_reg_models,psize): #Image should be YUV
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

if __name__ == "__main__":
    S = cv2.imread("img_color.jpg") # Source color image
    T = cv2.imread("img_gray.jpg",0) # Target gray image

    #--- YUV Conversion
    S = cv2.cvtColor(cv2.imread("img_color.jpg"),cv2.COLOR_BGR2YUV) 
    T = cv2.cvtColor(cv2.imread("img_gray.jpg",0),cv2.COLOR_GRAY2BGR) 
    T = cv2.cvtColor(T,cv2.COLOR_BGR2YUV)

    patches_filename = "patches_psize-"+str(PatchSize)+",pstride-"+str(PatchStride)
    try:
        (patches,u_vals,v_vals) = pickle.load( open( patches_filename+".p", "rb" ) )
        print "Successfully loaded pickle file:",patches_filename
    except IOError:
        print "Pickle file not found:",patches_filename,"\n Generating patches..."
        patches,u_vals,v_vals = PatchExtractor(S,PatchSize,PatchStride)
        pickle.dump( (patches,u_vals,v_vals), open( patches_filename+".p", "wb" ),protocol=pickle.HIGHEST_PROTOCOL )

    train_filename = "KnRmodel_n-clusters-"+str(NumClusters)+"_"+patches_filename

    try:
        (kmodel,u_reg_models,v_reg_models) = pickle.load( open( train_filename+".p", "rb" ) )
        print "Successfully loaded pickle file:",train_filename
    except IOError:
        print "Pickle file not found:",train_filename,"\n Training models..."
        kmodel,u_reg_models,v_reg_models = TrainKmeansAndRegression(patches,u_vals,v_vals)
        pickle.dump( (kmodel,u_reg_models,v_reg_models), open( train_filename+".p", "wb" ),protocol=pickle.HIGHEST_PROTOCOL )

    #--- Colorization step
    output_yuv = AssignColor(T,kmodel,u_reg_models,v_reg_models,PatchSize)

    output_image = cv2.cvtColor(output_yuv,cv2.COLOR_YUV2BGR)
    cv2.imwrite("ColorizedOutput.jpg",output_image)
    cv2.imshow("Final output",output_image)

    while cv2.waitKey(0) != 27:
        cv2.waitKey(10000)