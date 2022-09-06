import numpy as np
import scipy.linalg as sl


A = np.array([[10., 2., 1.],[6., 5., 4.],[1., 4., 7.]])
print(sl.norm(A))
# Forbenius Norm

print(sl.norm(A,2))
# which is defined as (!!!!)
print(np.sqrt(np.real((np.max(sl.eigvals( A.T @ A))))))
# i.e. involves the eigenvalues of the matrix A.T@A, 
# or the so-called "singular values" of A - see module on Geophysical Inversion

print(A)
print(np.linalg.cond(A))  # let's use the in-built condition number function
print(sl.norm(A,2)*sl.norm(sl.inv(A),2))  # so the default condition number uses the matrix two-norm
print(np.linalg.cond(A,'fro')) # this is how to use a different norm
print(sl.norm(A,'fro')*sl.norm(sl.inv(A),'fro'))



# Jacobi's method
A = np.array([[10., 2., 3., 5.],[1., 14., 6., 2.],[-1., 4., 16., -4],[5. ,4. ,3. ,11. ]])
b = np.array([1., 2., 3., 4.])

def Jacobi(A, b):                   # compute for solution x for ill-conditioned matrices
    x = np.zeros(A.shape[0]) 
    tol = 1.e-6 
    it_max = 1000
    residuals=[] 
    for it in range(it_max):
        x_new = np.zeros(A.shape[0])  # initialise the new solution vector
        for i in range(A.shape[0]):
            x_new[i] = (1./A[i, i]) * (b[i] 
                                       - (np.dot(A[i, :i], x[:i]) 
                                       + np.dot(A[i, i+1:], x[i+1:])))
        residual = sl.norm(A @ x - b)  # calculate the norm of the residual r=Ax-b for this latest guess
        residuals.append(residual) # store it for later plotting
        if (residual < tol): # if less than our required tolerance jump out of the iteration and end.
            break
        x = x_new # update old solution
    return x, residual


# gauss seidel
def gauss_seidel(A, b, maxit=500, tol=1.e-6):
    m, n = A.shape
    x = np.zeros(A.shape[0])
    residuals = []
    for k in range(maxit):
        for i in range(m):
            x[i] = (1./A[i, i]) * (b[i] 
                                   - np.dot(A[i,:i], x[:i]) 
                                   - np.dot(A[i,i+1:], x[i+1:]))     # instantly update new x in calculation
        residual = sl.norm(A@x - b)
        residuals.append(residual)
        if (residual < tol): break    
    return x, residuals