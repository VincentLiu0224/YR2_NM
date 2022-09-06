import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import scipy.stats as ss
import newton_new as nn               # newton polynomial, use func_newton(xDATA, yDATA, x*)

x = np.array([1.0, 2.0, 5.0, 6.0, 7.0])  #x value
x2 = np.array([1.0, -2.0, 3.0, -4.0, 5.0])   #y value
yi = np.array([1.0, 8.0, 27.0])
xi = np.linspace(0.2, 7.0, 100)


def func(x):
    return x**3


def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'ko', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


def func2(t):
    return 1 - np.exp((-9*(10**-3)*t)/(2*(0.001**2)*2650))


# set up figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))

# For clarity we are going to add a small margin to all the plots.
ax1.margins(0.1)


degree = 3                                     # degree of polynomial, the highest power of x/or curve-fitting with lower power i.e.1
coeff = np.polyfit(x, x2, degree)              # least square fitting
print(coeff)                                   # returns degree+1 number in list with highest power first                         
p = np.poly1d(coeff)                           # make coeff as a function


o = sci.lagrange(x, x2)                        # lagrange polynomial, returns function


ax1.plot(xi, p(xi))
ax1.plot(xi, o(xi))


error = 0
for i in range(4):                             # square error of predicted 'p func list' and 'exact x2 list'
    error += (p(x[i])-x2[i])**2
print(error) 


plot_raw_data(x, x2, ax1)                      # plot the raw data



# add a figure title
ax1.set_title('Our simple raw data', fontsize=16)

# Add a legend
ax1.legend(loc='upper left', fontsize=18)

file = open("Length-Width.dat", 'r')

x3 = []
y3 = []
for line in file:
    x3.append(float(line.split()[0]))
    y3.append(float(line.split()[1]))
x0 = np.linspace(-0.5, 2.5, 100)
x4 = np.log10(np.array(x3))
y4 = np.log10(np.array(y3))



# app = sci.lagrange(x3, y3)
coefficients = np.polyfit(x4, y4, 1)
pred = np.poly1d(coefficients)
ax2.plot(x0, pred(x0))
# ax2.plot(x0, app(x0))
plot_raw_data(x4, y4, ax2)



SS_res = np.sum((pred(x4) - y4)**2)
SS_tot = np.sum((np.mean(y4) - y4)**2)
r2 = 1. - SS_res/SS_tot
print(r2)                                        # The r squared error


t = np.linspace(0, 5, 100)
ax3.plot(t, func2(t))



plt.show()
