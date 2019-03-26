import numpy as np


def gini(distribution):
    """
    Calculates gini impurity for the given distribution
    """
    sample_count = length(distribution)
    # Get the count of each unique value in distribution.
    _, per_class_count = np.unique(distribution, return_counts = True)

    # Calculate the gini impurity.
    gini_impurity = 1 - sum(np.square(per_class_count))/(sample_count * sample_count)

    return gini_impurity


def gini_boundary(X, Y, minleaf = 1):
	cut_value = 0
	impurity  = 100
	M         = len(Y)
	Labels, frequency = np.unique(Y, return_counts=True)    

	frequency_right   = frequency
	frequency_left    = np.zeros(len(Labels),)

	pre_gini  = gini(Y)



	sorted_values   = np.sort(X, 'ascend')
	sorted_indices  = np.argsort(X, 'ascend')
	sorted_labels   = Y[sorted_indices]



	j = minleaf #Discuss Index Issues

	while(j<=M-minleaf):

		labels                      = sorted_labels[j:]
		labels                      = sorted_labels[sorted_values[j:] == sorted_values[j]]
		labels_, counts     	    = np.unique(labels, return_counts = True)



		frequency_right[labels_]   -= counts
		frequency_left[labels_]    += counts


		gl = np.dot(frequency_left, frequency_left)
		gr = np.dot(frequency_right, frequency_right)

		j += len(labels)

		if j<= M-minleaf:
			gl        = 1 - float(gl)/((j-1)*(j-1))
			gr        = 1 - float(gr)/((j-1)*(j-1))
			post_gini = (j-1)*gl/M + (M-j+1)*gr/M

		else:
			post_gini = pre_gini + 1


		if post_gini < pre_gini:
			pre_gini  = post_gini
			cut_value = 0.5*(sorted_values[j] + sorted_values[j-1])
			impurity  = post_gini

		return cut_value, impurity








		





















