#########################################################################
# TrainKmeansAndRegression:
#   Trains the K-means clustering model and uses
#   the U-V values of the patch centers to train
#   regression models for each cluster center
#########################################################################
from dependency import *

class TrainKmeansAndRegression:
    def __init__(self,patches,u_vals,v_vals):
        self.kmodel,self.u_reg_models,self.v_reg_models = self.function(patches,u_vals,v_vals)


def function(x,patches,u_vals,v_vals):
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