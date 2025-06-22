import numpy as np

def constructingFishermatrix(data_vectors, covariance_matrix, dtheta, n_params_p_m, n_params_p=0):
    """
    Constructs the Fisher matrix from data vectors and a covariance matrix.

    Parameters:
    data_vectors (list of numpy arrays): The data vectors for which the Fisher matrix is to be constructed. The ith element is an array containing two vectors:
        - The first vector corresponds to the simulation for parameter p.
        - The second vector corresponds to the simulation for parameter m OR The second vector corresponds to the simulation for fiducial parameters.
        Also, if n_params_p!=0, all such parameters, must necessarily come after the parameters with both simulations for p and m.
    covariance_matrix (numpy array): The covariance matrix associated with the data vectors.
    n_params_p_m (int): The number of parameters with both simulations for p and m.
    n_params_p (int): The number of parameters with just simulations for p.
    dtheta (list of float): The parameter step sizes for the derivatives. Also, if n_params_p!=0, all such parameters, must necessarily come after the parameters
    with both simulations for p and m.
    n (int): The number of realizations.

    Returns:
    numpy array: The constructed Fisher matrix. #Give the expression

    Raises ValueError:
    -If the length of data_vectors are not of equal length
    -If the covariance matrix is not square or does not match the length of data_vectors
    - If length of dtheta and n_params_p_m + n_params_p do not match
    """
    # Checking all the input parameters
    p=len(data_vectors[0][0])
    for i in range(len(data_vectors)):
        for j in range(2):
            if len(data_vectors[i][j]) != p:
                raise ValueError("All data vectors must be of equal length.")
    if np.shape(covariance_matrix)[0] != np.shape(covariance_matrix)[1]:
        raise ValueError("Covariance matrix must be square.")
    if np.shape(covariance_matrix)[0] != len(data_vectors):
        raise ValueError("Covariance matrix must match the length of data vectors.")
    if len(dtheta) != n_params_p_m + n_params_p:
        raise ValueError("Length of dtheta must match n_params_p_m + n_params_p.")
    # Constructing the derivatives
    d = np.zeros([n_params_p_m+n_params_p,len(data_vectors)])
    for i in range(n_params_p_m):

        d_p = data_vectors[i][0]
        d_m = data_vectors[i][1]
        delt = d_p - d_m
        d[i] = delt/(2*dtheta[i])

    if n_params_p > 0:
        for j in range(n_params_p_m, n_params_p_m + n_params_p):
            d_p= data_vectors[i][0]
            d_0= data_vectors[i][1]
            delt = d_p - d_0
            d[j] =  delt/(dtheta[j])
    
    # The Fisher matrix
    c_inv=np.linalg.inv(covariance_matrix)
    F = np.zeros([n_params_p+n_params_p_m,n_params_p_m+n_params_p])
    for i in range(0, n_params_p_m + n_params_p):
        for j in range(0, n_params_p_m + n_params_p):
            F[i][j] = (np.transpose(d[i])).dot(c_inv).dot(d[j])