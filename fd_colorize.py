#!/usr/bin/python

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
# from sklearn.feature_extraction.image import PatchExtractor

PatchSize = 27

# PE = PatchExtractor(patch_size=(13,13),max_patches=0.01,random_state=0)
# T1 = np.expand_dims(T,axis=0)
# print T1.shape
# PE.fit(T1)
# patches = PE.transform(T1)


# for i in range(patches.shape[0]):
# 	tmp = patches[i][:]
# 	print "Tmp:",tmp.shape
# 	cv2.imshow("Tmp",cv2.resize(tmp,dsize=(640,640)) )
# 	cv2.waitKey(100)

def PatchExtractor(image,psize,stride=None):
	patches = []
	if stride is None:
		stride = psize

	dim = 0 # Use only Y (Luminance)
	
	# for dim in range(image.shape[2]):
	for nx in range( (1+image.shape[0]-psize)//stride ):
		for ny in range( (1+image.shape[0]-psize)//stride ):
			# patches.append(image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim])
			patch = image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim]
			# print patch.reshape((-1,1)).shape
			patches.append( patch.reshape((1,-1)) )
	return np.vstack(patches)

# cv2.imshow("S",S)
# cv2.imshow("T",T)

# while cv2.waitKey(0) != 27:
# 	pass

if __name__ == "__main__":
	S = cv2.imread("img_color.jpg")
	T = cv2.imread("img_gray.jpg",0)

	S = cv2.cvtColor(cv2.imread("img_color.jpg"),cv2.COLOR_BGR2YUV) # Source color image
	T = cv2.cvtColor(cv2.imread("img_gray.jpg",0),cv2.COLOR_GRAY2BGR) # Target gray scale image
	T = cv2.cvtColor(T,cv2.COLOR_BGR2YUV) # Target gray scale imagek

	patches = PatchExtractor(S,PatchSize)
	print "Patches shape:",patches.shape

	kmodel = KMeans(n_clusters=27,n_jobs=-1)
	kmodel.fit(patches)
	print "Model fit done! "

	mean_patch = np.zeros((PatchSize,PatchSize))
	for pt in kmodel.cluster_centers_ :
		mean_patch = pt.reshape(PatchSize,PatchSize)
		mean_patch = np.uint8(mean_patch)
		mean_patch = cv2.cvtColor(mean_patch,cv2.COLOR_GRAY2BGR)
		# print mean_patch
		cv2.imshow("Mean patch",mean_patch)
		cv2.waitKey(0)
