import numpy as np

# After smoothing x,y,z, extract vector-magnitude of smoothed x,y,z / Returns norm of given vectors (list)
def get_norm(*columns):
    # columns : is a package of np.arrays (np.arrays of X, Y ,Z data)
    # Index: 0,1,2 = X, Y, Z

    if len(columns) != 3:
        print('get_norm() Error: must input three parameters')
        pass

    # zip X, Y, Z data together, make it an element
    vector_list = list(zip(columns[0], columns[1], columns[2]))  # zip x,y,z values and store them in a list
    # get vector magnitude, store them in a list
    norm = [np.linalg.norm(vec) for vec in vector_list]
    return norm # an element's type = numpy.float64

# Return mean, std, sum of an array
def extract_feat_mss(arr:np.array)->tuple:
    if not type(arr) is np.array:
        arr = np.array(arr)
    return arr.mean(), arr.std(), arr.sum()

#
def get_corr(X_data_fil, Y_data_fil, Z_data_fil)->dict:
    '''
    Return corr_coef of 3-axis data. Each axis data has to be already-filtered.
    :param X_data_fil: 
    :param Y_data_fil: 
    :param Z_data_fil: 
    :return: 
    '''
    # params: two variables that we want to find the correlation coefficient
    corr_XY = np.corrcoef(X_data_fil, Y_data_fil)[0, 1]
    corr_YZ = np.corrcoef(Y_data_fil, Z_data_fil)[0, 1]
    corr_ZX = np.corrcoef(Z_data_fil, X_data_fil)[0, 1]

    return {'corr XY': corr_XY, 'corr YZ': corr_YZ, 'corr ZX': corr_ZX}

