import numpy as np
from getdist.gaussian_mixtures import GaussianND


def constructingFishermatrix(cov_matrix, der_matrix, n_cov, mean, labels):
    """
    Caluclated the Fisher information matrix given the covariance matrix and the derivative matrix. The covariance matrix is the covariance of the data, and the derivative matrix is the derivative of the data with respect to the parameters. The Fisher information matrix is given by the formula:

    Inputs:
    cov_matrix: The covariance matrix of the data. This should be a square matrix of shape (n, n), where n is the number of data points.
    der_matrix: The derivative of the data with respect to the parameters. This should be a matrix of shape (m, n), where n is the number of data points and m is the number of parameters.
    n_cov: The number of realizations the covariance matrix is estimated from.

    Returns:
    fisher_matrix: The Fisher information matrix. This will be a square matrix of shape (m, m), where m is the number of parameters.
    """
    # Calculating the hartlap factor
    hartlap = (n_cov - cov_matrix.shape[0] - 2) / (n_cov - 1)
    c_inv = hartlap*np.linalg.inv(cov_matrix)
    F = np.zeros([der_matrix.shape[0], der_matrix.shape[0]])
    for i in range(der_matrix.shape[0]):
        for j in range(der_matrix.shape[0]):
            F[i][j] = (np.transpose(der_matrix[i])).dot(c_inv).dot(der_matrix[j])

    # Build an N-D Gaussian for the parameters
    cov=np.linalg.inv(F)
    gauss = GaussianND(mean, cov, labels=labels)

    return F, gauss
