import numpy as np
import scipy.linalg as sl

# partial pivoting
def swap(A, b, roll1):
    A_copy = A.copy()
    columns = A[roll1:, roll1]
    index = np.argmax(np.abs(columns))
    be_swaped = [A_copy[roll1, :], b[roll1]]
    to_swap = [A_copy[index+roll1, :], b[index+roll1]]
    A[roll1, :] = to_swap[0]
    A[index+roll1, :] = be_swaped[0]
    b[roll1] = to_swap[1]
    b[index+roll1] = be_swaped[1]
    return A, b

# LU decomposite method, order=3
def upper_lower_trangle(A, b):
    roll, column = np.shape(A)
    n = np.size(b)
    template1 = A.copy()
    template2 = np.eye(n)
    assert roll == n and column == n
    for i in range(n-1):
        template1, b = swap(template1, b, i)
        for k in range(i+1, n):
            c = template1[k, i]/template1[i, i]
            template1[k] = -(c*template1[i, :]-template1[k, :])
            template2[k, i] = c
            #b[k] = -c*b[i]-b[k]
    return template1, template2, b

# solve LU decomposite: L*c=b----U*x=c
def solve(upper, lower, b):
    n = np.size(b)
    for i in range(0, n):
        b[i] = b[i] - np.dot(lower[i, :i], b[:i])
    b[-1] = b[-1]/upper[-1, -1]
    for k in range(2, n+1):
        b[-k] = (b[-k]-np.dot(upper[-k, -k+1:], b[-k+1:]))/upper[-k, -k]
    return b

 
A = np.array([[5.0, 7.0, 5.0, 9.0], [5.0, 14.0, 7.0, 10.0], [20.0, 77.0, 41.0, 48.0], [25.0, 91.0, 55.0, 67.0]])
b = np.array([10.0, 20.0, 30.0, 40.0])
print(sl.inv(A)@b)
upper, lower, b = upper_lower_trangle(A, b)
print(lower)
print(solve(upper, lower, b))


# (np.dot(lower[2, :2], b[:2]))
