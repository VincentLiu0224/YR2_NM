import numpy as np
import scipy.linalg as sl


# forward and backward subsitution, order=2
def upper_trangle(A, b):
    roll, column = np.shape(A)
    n = np.size(b)
    assert roll == n and column == n
    for i in range(n-1):
        for k in range(i+1, n):
            c1, c2 = A[k, i], A[i, i]
            A[k] = A[k,:]-(c1/c2)*A[i,:]
            b[k] = b[k]-(c1/c2)*b[i]
    b = back_sub(A, b, n)
    return A, b


def back_sub(A, b, n):
    for i in range(1, n+1):
        b[-i] = b[-i]/A[-i, -i]
        # A[-n-1, -n-1] = 1
        for k in range(i+1, n+1):
            b[-k] = b[-k]-b[-i]*A[-k, -i]
    return b


A = np.array([[10., 2., 1.],[6., 5., 4.],[1., 4., 7.]])
# the total size of the array storing A - here 9 for a 3x3 matrix
print(np.size(A)) 
# the number of dimensions of the matrix A
print(np.ndim(A))
# the shape of the matrix A
print(np.shape(A))
# and so if we need the number of rows say:
print('The number of rows in A is', np.shape(A)[0])
# the transpose of the matrix A
print(A.T)
# the inverse of the matrix A - computed using a scipy algorithm
print(sl.inv(A))
# the determinant of the matrix A - computed using a scipy algorithm
print(sl.det(A))
# Multiply A with its inverse using the @ matrix multiplication operator. 
# Note that due to roundoff errors the off diagonal values are not exactly zero.
print(A @ sl.inv(A))
# same way to achieve the same thing
print(A.dot(sl.inv(A)))
# the @ operator is realtively new so you may still see use of dot in some code/courses
# note that the * operator simply does operations element-wise - here this
# is not what we want!
print(A*sl.inv(A))
# if you're familiar with Matlab this poitwise operation is what you get with dot-star: ".*"
# how to initialise a vector of zeros 
print(np.zeros(3))
# how to initialise a matrix of zeros 
print(np.zeros((3,3)))
# how to initialise a 3rd-order tensor of zeros
print(np.zeros((3,3,3)))
# shortcut to create an array of zero of the same size as A
B = np.zeros_like(A)
print(B)
# how to initialise the identity matrix, I or Id
print(np.eye(3))


A = np.array([[2., 3., -4.], [6., 8., 2.], [4., 8., -6.]])
b = np.array([5., 3., 19.])
print(sl.inv(A)@b)
print(upper_trangle(A, b))

