import numpy as np
import swap_engine as sp


def gaussPivot(a, b, tol=1.0e-12):
    n = len(b)
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i, :]))
    for k in range(0, n-1):
        p = np.argmax(np.abs(a[k:n, k])/s[k:n]) + k
        try:
            singularity = bool(abs(a[p, k]) < tol)
        except singularity is True:
            print("matrix is singular")
            break
        else:
            if p != k:
                sp.swapRows(b, k, p)
                sp.swapRows(s, k, p)
                sp.swapRows(a, k, p)
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k]/a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam*a[k, k+1:n]
                b[i] = b[i] - lam*b[k]
    try:
        singularity2 = bool(abs(a[n-1, n-1]) < tol)
    except singularity2 is True:
        print("matrix is singular")
    else:
        b[n-1] = b[n-1]/a[n-1, n-1]
        for k in range(n-2, -1, -1):
            b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1:n]))/a[k, k]
    return b
