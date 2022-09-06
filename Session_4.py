import string
import numpy as np
import rootsearch as rs
import newton_raphson_new as nr
import rootsearch_modified as rsm
import scipy.optimize as sop
import matplotlib.pyplot as plt


def picard_bracketing(f, xARRAY):
    solution_list = []
    for i in xARRAY:
        solu = picard(f, i)
        try:
            assert solu <= xARRAY[-1] and solu >= xARRAY[0]
        except AssertionError:
            continue
        else:
            try:
                assert len(solution_list) == 0
            except AssertionError:
                for i in solution_list:
                    if abs(solu-i) >= 1e-3:
                        solution_list.append(solu)
                    else:
                        continue
            else:
                solution_list.append(solu)
    return solution_list


def picard(f, x, **kwargs):       # by finding g(x)=x to find the root, need to convert
    func = x + 0.0001
    iteration = 0              # to g(x)=x format
    while abs(func-x) >= 1e-5:
        x = func
        func = f(x)
        iteration += 1.0
        if iteration >= 30:
            # x = "too much iteration"
            return 1e26
    return x


def picard_course(f, x, atol=1.0e-6):
    fevals = 0
    x_prev = x + 2*atol
    while abs(x - x_prev) > atol:
        x_prev = x
        x = f(x_prev)
        fevals += 1
        print('Current iteration solution: ',x)
    print('\nPicard used', fevals, 'function evaluations')
    return x




def graphic_rootsearch(f, xARRAY):
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    ax1.plot(xARRAY, f(xARRAY), 'b', label='$y(x) = f(x) = x - \mathrm{e}^{-x}$')
    # add a zero line extending across axes
    xlim = ax1.get_xlim()
    ax1.plot([xlim[0], xlim[1]], [0., 0.], 'k--', label='$y(x)=0$')
    ax1.plot(0.5671, 0., 'ro', label='intersection')
    ax1.set_xlim(xlim)
    ax1.legend(loc='best', fontsize=14)
    ax1.set_xlabel('$x$', fontsize=16)
    ax1.set_ylabel('$y(x)$', fontsize=16)
    ax1.set_title('Fixed point as a root of $f(x)$', fontsize=16)
    plt.show()


def plot_root_bracketing(f, a, b, dx, ax, xbounds=(-0.1, 1.4), ybounds=(-5, 6), flabel=''):
    x = np.linspace(a, b, int((b-a)/dx)+1)
    y = f(x)
    # plot the sub-intervals in blue
    ax.plot(x, y, 'bo-')
    for i in range(1, len(x)):
        if np.sign(y[i]) != np.sign(y[i-1]):
            # plot the sub-interval where the sign changes in red
            ax.plot([x[i], x[i-1]], [y[i], y[i-1]], 'ro-')
    ax.set_xlabel('$x$', fontsize=16)
    if not flabel:
        fl = '$f(x)$'
    else:
        fl = flabel
    ax.set_ylabel(fl, fontsize=16)
    xlim = ax.get_xlim()
    ax.plot([xlim[0], xlim[1]], [0., 0.], 'k--')
    ax.set_xlim(xlim)
    ax.set_title('Root bracketing\n' + '(red indicates the bracket containing the root)', fontsize=20)
    plt.show()


def f(x):
    return x**3 + x**2 - 10*x



# print(picard_bracketing(f, np.linspace(-4.0, 2.0, 10000)))
# fig, axl = plt.subplots(figsize=(8, 5))
# plot_root_bracketing(f, -10, 10, 0.5, axl)
print(rs.main_bisection(f, -100, 100))
print(rsm.main(f, -100, 100))
# print(sop.bisect(f, 0, 5))
# print(nr.ne_raph(f, df, -1))
# print(nr.ne_raph_diff_secant(f, 0, 0.1))

