#!/usr/bin/python

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

PatchSize = 27
PatchStride = PatchSize

def PatchExtractor(image,psize,stride=None):
	assert(psize%2 == 1) # Patch size must be odd

	patches = []
	# ytrain = np.zeros( (image.shape[0] - (1+psize),image.shape[1] - (1+psize)) )
	ytrain = np.zeros(( (1+image.shape[0]-psize)//stride, (1+image.shape[1]-psize)//stride ))
	if stride is None:
		stride = psize

	dim = 0 # Use only Y (Luminance)
	
	# for dim in range(image.shape[2]):
	for ny in range( (1+image.shape[1]-psize)//stride ):
		for nx in range( (1+image.shape[0]-psize)//stride ):
			# patches.append(image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim])
			patch = image[nx*stride:(nx*stride)+psize,ny*stride:(ny*stride)+psize,dim]
			# print patch.reshape((-1,1)).shape
			patches.append( patch.reshape((1,-1)))
			ytrain[nx,ny] = image[(nx*stride)+(psize//2),(ny*stride)+(psize//2),dim]

	ytrain = ytrain.reshape((-1,1))
	return np.vstack(patches),ytrain

# while cv2.waitKey(0) != 27:
# 	pass

if __name__ == "__main__":
	S = cv2.imread("img_color.jpg") # Source color image
	T = cv2.imread("img_gray.jpg",0) # Target gray image

	# YUV Conversion
	S = cv2.cvtColor(cv2.imread("img_color.jpg"),cv2.COLOR_BGR2YUV) 
	T = cv2.cvtColor(cv2.imread("img_gray.jpg",0),cv2.COLOR_GRAY2BGR) 
	T = cv2.cvtColor(T,cv2.COLOR_BGR2YUV)

	patches,ytrain = PatchExtractor(S,PatchSize,PatchStride)
	print "Patches shape:",patches.shape

	lrmodel = LinearRegression(n_jobs=-1)
	kmodel = KMeans(n_clusters=3,n_jobs=-1)
	labels = kmodel.fit_predict(patches)
	print "Model fit done! "

	mean_patch = np.zeros((PatchSize,PatchSize))
	for i,pt in enumerate(kmodel.cluster_centers_):
		mean_patch = pt.reshape(PatchSize,PatchSize)
		mean_patch = np.uint8(mean_patch)
		mean_patch = cv2.cvtColor(mean_patch,cv2.COLOR_GRAY2BGR)
		# print mean_patch
		# cv2.imshow("Mean patch",mean_patch)
		# cv2.waitKey(0)
		nearest_points = patches[labels == i,]
		y_regress = ytrain[labels == i]

		# To verify that average of the spliced points is actually the mean_patch, use this line:
		# print "Average: ",np.average(nearest_points,axis=0),"\nvs mean:",pt

		print "nearest_points:",nearest_points.shape
		print "ytrain:",y_regress.shape
		lrmodel.fit(nearest_points,y_regress)

