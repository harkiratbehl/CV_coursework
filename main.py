from dependency import *
#--- Global declarations
NumClusters = 20
PatchSize = 15
PatchStride = PatchSize

if __name__ == "__main__":
    S = cv2.imread("img_color.jpg") # Source color image
    T = cv2.imread("img_gray.jpg",0) # Target gray image

    #--- YUV Conversion
    B=cv2.imread("img_color.jpg")
    S = cv2.cvtColor(B,cv2.COLOR_BGR2YUV)
    A=cv2.imread("img_gray.jpg",0)
    # cv2.imshow('gray_grayimage',A) 
    T = cv2.cvtColor(A,cv2.COLOR_GRAY2BGR) 
    # cv2.imshow('bgr_grayimage',T)
    T = cv2.cvtColor(T,cv2.COLOR_BGR2YUV)
    # cv2.waitKey(0)

    patches_filename = "patches_psize-"+str(PatchSize)+",pstride-"+str(PatchStride)
    try:
        (patches,u_vals,v_vals) = pickle.load( open( patches_filename+".p", "rb" ) )
        print "Successfully loaded pickle file:",patches_filename
    except IOError:
        print "Pickle file not found:",patches_filename,"\n Generating patches..."
        a = patch_generator(S,PatchSize,PatchStride)
        patches,u_vals,v_vals = a.patches,a.u_vals,a.v_vals
        pickle.dump( (patches,u_vals,v_vals), open( patches_filename+".p", "wb" ),protocol=pickle.HIGHEST_PROTOCOL )

    train_filename = "KnRmodel_n-clusters-"+str(NumClusters)+"_"+patches_filename

    try:
        (kmodel,u_reg_models,v_reg_models) = pickle.load( open( train_filename+".p", "rb" ) )
        print "Successfully loaded pickle file:",train_filename
    except IOError:
        print "Pickle file not found:",train_filename,"\n Training models..."
        b = TrainKmeansAndRegression(patches,u_vals,v_vals)
        kmodel,u_reg_models,v_reg_models = b.kmodel,b.u_reg_models,b.v_reg_models
        pickle.dump( (kmodel,u_reg_models,v_reg_models), open( train_filename+".p", "wb" ),protocol=pickle.HIGHEST_PROTOCOL )

    #--- Colorization step
    c = AssignColor(T,kmodel,u_reg_models,v_reg_models,PatchSize)
    output_yuv = c.output_yuv
    output_image = cv2.cvtColor(output_yuv,cv2.COLOR_YUV2BGR)
    cv2.imwrite("ColorizedOutput.jpg",output_image)
    cv2.imshow("Final output",output_image)

    while cv2.waitKey(0) != 27:
        cv2.waitKey(10000)