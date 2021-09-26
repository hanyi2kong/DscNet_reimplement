import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


def thr_c(coe, alpha):
    if alpha < 1:
        n = coe.shape[1]
        cp = np.zeros((n, n))
        s = np.abs(np.sort(-np.abs(coe), axis=0))
        ind = np.argsort(-np.abs(coe), axis=0)

        for i in range(n):
            cl1 = np.sum(s[:, i]).astype(float)
            stop = False
            c_sum = 0
            t = 0
            while not stop:
                c_sum = c_sum + s[t, i]
                if c_sum > alpha * cl1:
                    stop = True
                    cp[ind[0:t + 1, i], i] = coe[ind[0:t + 1, i], i]
                t = t + 1
    else:
        cp = coe

    return cp


def post_proc(coe, k, d, ro, comment):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = coe.shape[0]
    coe = 0.5 * (coe + coe.T)
    if comment is False:
        coe = coe - np.diag(np.diag(coe)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * k + 1
    u, s, _ = svds(coe, r, v0=np.ones(n))
    u = u[:, ::-1]
    s = np.sqrt(s[::-1])
    s = np.diag(s)
    u = u.dot(s)
    u = normalize(u, norm='l2', axis=1)
    z = u.dot(u.T)
    z = z * (z > 0)
    l_vec = np.abs(z ** ro)
    l_vec = l_vec / l_vec.max()
    l_vec = 0.5 * (l_vec + l_vec.T)
    spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(l_vec)
    grp = spectral.fit_predict(l_vec)
    return grp, l_vec


def spectral_clustering(coe, k, d, alpha, ro, comment=True):
    coe = thr_c(coe, alpha)
    y, _ = post_proc(coe, k, d, ro, comment)
    return y
