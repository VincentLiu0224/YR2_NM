# central difference
import scipy.interpolate as si
import numpy as np
import matplotlib.pyplot as plt
import rootsearch_modified as rsm
import Session_4 as s4
import newton_raphson_new as nrn
import matplotlib.pyplot as plt

def central_diff(f, x, t, dx):                                    # central difference method with f, xDATA and defined dx
    return (f(x+dx, t)-f(x-dx, t))/(2*dx)                         # can be convert to a inte method, p=2


def discrete_points1(f, x, t, dx):                                # discrete point method/forward differential with f, xDATA and defined dx
    return (f(x+dx, t)-f(x, t))/dx

def backward_diff(f, x, t, dx):                                   # backward differential method from ML3
    return (f(x, t)-f(x-dx, t))/dx

def discrete_points2(yARRAY, xARRAY):
    diffs = []
    for i in range(len(xARRAY)):
        dy = yARRAY[i+1]-yARRAY[i]
        dx = xARRAY[i+1]-xARRAY[i]
        diffs.append(dy/dx)
    return diffs




def secondary_derivative1(f, x, t, dx):                            # secondary derivative with f, xDATA and defined dx
    return (f(x+dx, t)+f(x-dx, t)-f(x, t)*2)/(dx**2)


def secondary_derivative(yARRAY, xARRAY):                       
    diffs = []
    for i in range(1, len(xARRAY)):
        diff = yARRAY[i+1]-yARRAY[i-1]+2*yARRAY[i]/((xARRAY[i]-xARRAY[i-1])**2)
        diffs.append(diff)
    return diffs



# Error of integration scheme is equal to dt^p + h^q
# interdependance of dt and h

# The following u_diff is equivalent to f
def euler_diff(u_diff, u0, t0, tmax, dt):                         # forward euler's method; u0, t0 start values, with tmax                                                          
    u = [u0]                                                      # udiff is a given diff function of u
    t = [t0]                                                      # first order method (p=1)
    while t[-1] < tmax:                                           # trade-off between step size and accuracy
        u.append(u[-1] + u_diff(u[-1], t[-1])*dt)                 # Ut+1 = Ut + dt*f(t) 
        t.append(t[-1]+dt)
    return u, t


def simple_back_euler_diff(u_diff, u0, t0, tmax, dt):
    def R(u1, u0, t1, delta_t, f):
        return u1 - u0 - delta_t*f(u1, t1)
    u = [u0]                                                          # backward euler's method from ML_02, Ut+1 = Ut + dt*f(t+1)
    t = [t0]                                                          # computed to solve an equation:
    while t[-1] < tmax:                                               # R(Ut+1)= Ut+1 - Ut - dt*f(Ut+1, t+1)
        u_new = nrn.quasi_newton_for_backeu(R, u[-1], t[-1], dt, u_diff)
        u.append(u_new)
        t_new = t[-1] + dt
        t.append(t_new)
    return u, t
        
                                                                             


def heun(u_diff, u0, t0, tmax, dt):
    u = [u0]
    t = [t0]
    while t[-1] < tmax:
        ue = u[-1] + u_diff(u[-1], t[-1])*(dt)                                # Heun's method
        u.append(u[-1] + 0.5*dt*(u_diff(u[-1], t[-1])+u_diff(ue, t[-1]+dt)))  # predictor-corrector from Euler method
        t.append(t[-1]+dt)                                                    # Second order method (p = 2)
    return u, t                                                    


def expo(x):
    return np.sin(x)


def f_diff1(u, t):
    return np.cos(t)


def f(u, t):
    return u


# print(discrete_points(2.36, 2.37, 0.85866, 0.86289))

fig = plt.figure()
ax1 = fig.add_subplot(111)

f_2 = si.lagrange(np.array([0.1, 0.2, 0.3]), np.array([0.078348, 0.138910, 0.192916]))
print(central_diff(f_2, 0.2, 0.01))
dx_list = []
val1 = []
val2 = []
dx = 0.1
# for i in range(2):
    # val1.append(np.log10(np.cos(0.8)-discrete_points(expo, 0.8, dx)))
    # val2.append(np.log10(np.cos(0.8)-central_diff(expo, 0.8, dx)))
    # dx_list.append(dx)
    # dx = dx/10
# ax1.plot(dx_list, val1)
# ax1.plot(dx_list, val2)
# print(euler_diff(f_diff1, 0, 0, 10, 0.5))
ys, xs = simple_back_euler_diff(f_diff1, 0, 0, 10, 0.05)
yh, xh = heun(f_diff1, 0, 0, 10, 0.05)
print(ys)
print(yh)
plt.scatter(xs, ys)
plt.scatter(xh, yh)
plt.show()
#print(np.exp(10))
# plt.show()
