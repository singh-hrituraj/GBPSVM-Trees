import numpy as np
from scipy.linalg import eigh
from numpy.linalg import matrix_rank, inv, det, norm
import math
def gini_from_distribution(distribution):
    """
    Calculates gini impurity for the given distribution.
    """
    if len(distribution)==0:
        return -1
    sample_count = len(distribution)
    # Get the count of each unique value in distribution.
    _, per_class_count = np.unique(distribution, return_counts = True)

    # Calculate the gini impurity.  
    gini_impurity = 1 - sum(np.square(per_class_count))/(sample_count * sample_count)

    return gini_impurity
    

def gini_from_frequency(frequency):
    """
    Calculates gini impurity for the given frequency of a distribution. 
    """
    sample_count = sum(frequency)
    # Calculate the gini impurity.
    gini_impurity = 1 - sum(np.square(frequency))/(sample_count * sample_count)

    return gini_impurity


def find_best_split_gini(X, Y, minleaf):
    """
    Finds the feature and the value of the feature which produces the best
    split (into 2 groups) of X and Y. The best split is the one with minimum
    gini impurity.

    Parameters:
    X: Feature matrix of the data sample. 
       Expected shape: n_samples x n_features
    Y: Corresponding labels of the data samles.
       Expected shape: n_samples x 1
    minleaf: minimum number of samples required in either of the two
             split groups.

    Returns:
    best_ftr_idx : index of the feature which produces best split.
    best_cut_val : value of the feature which produces the best split.
    best_impurity: The gini impurity for the best split.
    """

    best_ftr_idx = 0
    best_cut_val = 0
    best_impurity = 100

    # Consider each feature separately and find the one that produces the best split.
    for ftr_idx in range(X.shape[1]):
        # Find the best cut value and impurity for this feature.
        cut_val, impurity = gini_boundary(X[:, ftr_idx], Y, minleaf)

        if (impurity < best_impurity):
            best_cut_val = cut_val
            best_ftr_idx = ftr_idx
            best_impurity = impurity

    return best_ftr_idx, best_cut_val, best_impurity


def gini_boundary(X, Y, minleaf = 1):
    """
    Helper function to find the cut value within X that produces minimum
    gini impurity. Note that here X contains only the values of one feature.
    """
    cut_value = 0
    impurity  = 100
    M         = len(Y)
    Labels, frequency = np.unique(Y, return_counts=True)    

    pre_gini  = gini_from_distribution(Y)

    sorted_values   = np.sort(X)
    sorted_indices  = np.argsort(X)
    sorted_labels   = Y[sorted_indices]

    j = minleaf # Discuss Index Issues

    # Initially assume the first j sorted points are on the left child of the root
    Labels_left, frequency_left_list = np.unique(sorted_labels[:j], return_counts = True)
    Labels_right, frequency_right_list = np.unique(sorted_labels[j:], return_counts = True)

    frequency_left = dict(zip(Labels_left, frequency_left_list))
    frequency_right = dict(zip(Labels_right, frequency_right_list))

    while(j<=M-minleaf-1):
        labels                      = sorted_labels[j:]
        labels                      = labels[sorted_values[j:] == sorted_values[j]]
        labels_, counts     	    = np.unique(labels, return_counts = True)
        
        # Transfer all these points from the right child to the left. 
        # We need to update the frequency of the labels in both nodes.
        for i in range(len(labels_)):
            frequency_right[labels_[i]] = frequency_right.get(labels_[i], 0) - counts[i]
            frequency_left[labels_[i]]  = frequency_left.get(labels_[i], 0) + counts[i]

        j += len(labels)

        if j <= M-minleaf:
            gl        = gini_from_frequency(list(frequency_left.values()))
            gr        = gini_from_frequency(list(frequency_right.values()))
            post_gini = (j-1)*gl/M + (M-j+1)*gr/M

        else:
            post_gini = pre_gini + 1

        if post_gini < pre_gini:
            pre_gini  = post_gini
            cut_value = 0.5*(sorted_values[j] + sorted_values[j-1])
            impurity  = post_gini

    return cut_value, impurity


def generalized_eigen_soln(G, H):
    """
    Solves the generalized eigen value problem for the matrix pair (G, H).
    Returns:
    eig_vals : list of eigen values in increasing order.
    eig_vectors: list of eigen vectors corresponding to the sorted list of eigen values.
    """
    eig_vals, eig_vectors = eigh(G, H)

    sorted_idx = np.argsort(eig_vals)

    return eig_vals[sorted_idx], eig_vectors[sorted_idx]


def selectHyperplane(X, Y, W, minleaf):
    """
    Selects the optimal hyperplane while using the output solution of 
    generalized eigen value problem on the bases of gini impurity 
    minimization
    """


    labels, frequency = np.unique(Y, return_counts=True)
    M                 = len(labels)

    pre_gini          = gini_from_distribution(Y)

    epsilon           = 0.1

    equal             = np.equal(W[:,0],W[:,1]).all()
    parallel          = np.dot(W[:,0],W[:,1])/(norm(W[:,0]*norm(W[:,1])))>1-epsilon


    if equal:
        W3 = W[:,0]
        W4 = W[:,1]

    elif parallel:
        W3 = W[:,0]
        W4 = W[:,1]

        W3[-1] = 0.5*(W3[-1] + W4[-1])
        W4[-1] = W3[-1]

    else:
        W3   = W[:,0]/norm(W[:-1,0]) + W[:,1]/norm(W[:-1,1])
        W4   = W[:,0]/norm(W[:-1,0]) - W[:,1]/norm(W[:-1,1])
    




    Y1 = np.dot(X,W3[:-1]) - W3[-1]
    Y2 = np.dot(X,W4[:-1]) - W4[-1]



    #New gini if we split using W3
    IndexPos      = np.nonzero(Y1>0)[0]
    IndexNeg      = np.nonzero(Y1<=0)[0]

    MinCriteriaW3 = (len(IndexPos) >= minleaf and len(IndexNeg) >= minleaf)

    labelsPos     = Y[IndexPos]
    labelsNeg     = Y[IndexNeg]
    giniPos       = gini_from_distribution(labelsPos)
    giniNeg       = gini_from_distribution(labelsNeg)

    PosRatio      = len(IndexPos)/float(len(Y))
    NegRatio      = len(IndexNeg)/float(len(Y))

    giniW3        = PosRatio*giniPos + NegRatio*giniNeg 


    #New Gini if we split using W4
    IndexPos      = np.nonzero(Y2>0)[0]
    IndexNeg      = np.nonzero(Y2<=0)[0]

    MinCriteriaW4 = (len(IndexPos) >= minleaf and len(IndexNeg) >= minleaf)
    


    labelsPos     = Y[IndexPos]
    labelsNeg     = Y[IndexNeg]
    

    giniPos       = gini_from_distribution(labelsPos)
    giniNeg       = gini_from_distribution(labelsNeg)

    PosRatio      = len(IndexPos)/float(len(Y))
    NegRatio      = len(IndexNeg)/float(len(Y))

    giniW4        = PosRatio*giniPos + NegRatio*giniNeg


    if MinCriteriaW3 and MinCriteriaW4:
        if giniW3 > giniW4:
            Plane = W4
        else:
            Plane = W3

        post_gini = min(giniW4, giniW3)
        
        if post_gini < pre_gini:
            splitFlag =  1
        else:
            splitFlag = -1

    else:
        if MinCriteriaW3:
            Plane    = W3
  
            if giniW3 < pre_gini:
                splitFlag = 1
            else:
                splitFlag = -1
        
        elif MinCriteriaW4:
            Plane    = W4
            if giniW4 < pre_gini:
                splitFlag = 1
            else:
                splitFlag = -1

        else:
            Plane     = W3
            splitFlag = -1
    return splitFlag, Plane


def is_singular(M):
    """
    Returns true if square matrix M is singular, else false.
    """
    return matrix_rank(M) != M.shape[0]


def bhattacharya_distance(x1_mean, x2_mean, x1_cov, x2_cov):
    """
    Calculates the Bhattacharya distance between two classes.
    Parameters:
    x1_mean, x2_mean: Mean of the two classes.
    x1_cov, x2_cov: Covariance matrices of the two classes.
    """

    if is_singular(x1_cov + x2_cov) or x1_cov.shape[0] == 1 or is_singular(x1_cov) or is_singular(x2_cov):
        return norm(x1_mean - x2_mean)

    # A temporary col vector to store differnce in mean. 
    temp = np.expand_dims(x1_mean - x2_mean, axis = 1)

    d = np.matmul(temp.T, inv((x1_cov + x2_cov)/2))
    d = np.matmul(d, temp)/8
    d += 0.5* np.log(det((x1_cov + x2_cov)/2) / math.sqrt(det(x1_cov)*det(x2_cov)))

    return d


def group(X, Y):
    """"
    For a given sample set with more than two classes, groups them into
    two hyper classes based on Bhattacharya distance.
    Parameters:
    X: Feature matrix.
    Y: Label matrix.
    """
    group1 = []
    group2 = []

    unique_labels = np.unique(Y)
    n_unique_labels = len(unique_labels)
    # Lists to store mean and covariance of each feature in X, per unique
    # label.
    x_mean = []
    x_cov  = []

    for label in unique_labels:
        # Get all the samples with this label.
        x_temp = X[Y==label, :]
        # Store mean and covariance.
        x_mean.append(np.mean(x_temp, axis = 0))
        if len(x_temp)==1:
            x_cov.append(np.zeros((len(x_temp[0]), len(x_temp[0]))))
        else:
            x_cov.append(np.cov(x_temp.transpose()))

    # Matrix to store the bhattacharya distance between every label pair.
    distance = np.zeros((n_unique_labels, n_unique_labels))

    # Populate the distance matrix.
    max_dist = -1
    max_dist_pair = []
    for i in range(n_unique_labels):
        for j in range(i+1, n_unique_labels):
            distance[i,j] = bhattacharya_distance(x_mean[i], x_mean[j], x_cov[i], x_cov[j])

            # Update max distance if required.
            if(distance[i,j] > max_dist):
                max_dist = distance[i,j]
                max_dist_pair = [i, j]

    # Since this matrix is symmetric, populate the lower triangle from the 
    # upper triangle values.
    # Get all the indices corresponding to the lower triangle.
    idx_lower_tri = np.tril_indices(distance.shape[0], -1)
    # Populate lower triangle
    distance[idx_lower_tri] = np.transpose(distance)[idx_lower_tri]
    
    # Put the labels with maximum distance in two separate groups.
    
    group1.append(unique_labels[max_dist_pair[0]])
    group2.append(unique_labels[max_dist_pair[1]])

    # For other labels, assign them the same group as the label in
    # max_dist_pair to which it is closer to.
    for i in range(n_unique_labels):
        if i in max_dist_pair:
            continue

        if distance[i, max_dist_pair[0]] <= distance[i, max_dist_pair[1]]:
            group1.append(unique_labels[i])
        else:
            group2.append(unique_labels[i])

    return group1, group2



