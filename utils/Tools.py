import numpy as np
import os

"""
    https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html?highlight=multivariate_normal#numpy.random.multivariate_normal
"""

def norm2d(mean, cov, size):
    ans = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=size,
        check_valid="raise")
    return ans.T

"""
make path, do not make file
"""
def genPath(*args):
    sep = "/"
    concat = sep.join(args)
    return concat

def mkPath(*args):
    tmp = genPath(args)
    if(not os.path.exists(tmp)):
        os.makedirs(tmp)
    return tmp

if __name__ == "__main__":
    # cov 为协方差矩阵
    ans = norm2d([0, 0], [[1, 0.3], [0.3, 4]], 10)
    print()
    pass
