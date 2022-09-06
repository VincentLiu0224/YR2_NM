import numpy as np


# middle point rule, twice as accurate as trapizoid rule
def mid_point_inti(f, xARRAY):
    inti = 0
    for i in range(len(xARRAY)-1):
        inti += f((xARRAY[i+1]-xARRAY[i])/2.0)*(xARRAY[i+1]-xARRAY[i])
    return inti

def midpoint_rule(a, b, function, number_intervals=10):
    interval_size = (b - a)/number_intervals
    assert interval_size > 0
    assert type(number_intervals) == int
    I_M = 0.0
    mid = a + (interval_size/2.0)
    while (mid < b):
        I_M += interval_size * function(mid)
        mid += interval_size
    return I_M


# trapizoid rule, two times less accurate as mid point rule
def trapizoid_inti(f, xARRAY):
    inti = 0
    for i in range(len(xARRAY)-1):
        inti += (f(xARRAY[i+1])+f(xARRAY[i]))*(xARRAY[i+1]-xARRAY[i])/2
    return inti

def trap_comp_inti(f, xARRAY):                                           # assume interval lengths are equal
    dx = xARRAY[1]-xARRAY[0]
    f_num = len(xARRAY)
    f_array = np.zeros(f_num)
    for i in range(f_num):
        f_array[i] = f(xARRAY[i])
    return (f_array[0]+f_array[-1]+2*sum(f_array[1:-1]))*dx/2


def trapezoidal_rule(a, b, function, number_intervals=10):             
    interval_size = (b - a)/number_intervals
    assert interval_size > 0
    assert type(number_intervals) == int
    I_T = 0.0
    for i in range(number_intervals):
        this_bin_start = a + (interval_size * i)
        I_T += interval_size * \
                (function(this_bin_start)+function(this_bin_start+interval_size))/2.0
    return I_T





# simpson rule of intigration, my version. assume inti = (x2-x1)*(f(x1)+f(x2)+4*f(x3))/6.0
# The errors are lower than for the midpoint and trapezoidal rules(order 4), and the method converge more rapidly 
# It is equivalent to r-k intogration is the function is only related to 1 varibale
def simpson_rule_inti(f, a1, a2, interval_num):
    inti = 0
    xARRAY = np.linspace(a1, a2, interval_num)
    for i in range(len(xARRAY)-1):
        x1, x2 = xARRAY[i], xARRAY[i+1]
        x3 = (x1 + x2)/2.0
        inti += (x2-x1)*(f(x1)+f(x2)+4*f(x3))/6.0
    return inti

def composite_sim_inti(f, x1, x2, interval_num):              # composite simpson rule, another compilation method of
    assert interval_num % 2 == 0                              # simpson's rule, faster and no difference
    interval = (x2 - x1)/interval_num
    I_cS2 = f(x1) + f(x2)
    for i in range(1, interval_num, 2):
        I_cS2 += f(x1 + interval*i)*4
    for i in range(2, interval_num-1, 2):
        I_cS2 += f(x1 + interval*i)*2
    return I_cS2*(interval/3.0)

def weddle_rule_inti(f, x1, x2, interval_num_1x):                    # weddle's rule, more accurate than simp's rule
    I_S_1x = simpson_rule_inti(f, x1, x2, interval_num_1x)
    I_S_2x = simpson_rule_inti(f, x1, x2, 2*interval_num_1x)
    return I_S_2x + ((I_S_2x + I_S_1x)/15)


# course version of inti rules
# if function depends on one variable, 3-steps Simpson's rule is equivalent to runge-kutta method
def simpsons_rule(a, b, function, number_intervals=10):
    interval_size = (b - a)/number_intervals
    assert interval_size > 0
    assert type(number_intervals) == int
    I_S = 0.0
    for i in range(number_intervals):
        this_bin_start = a + interval_size * (i)
        this_bin_mid = this_bin_start + interval_size/2
        this_bin_end = this_bin_start + interval_size
        I_S += (interval_size/6) * (function(this_bin_start) +
                                  4 * function(this_bin_mid) + function(this_bin_end))
    return I_S

def simpsons_composite_rule(a, b, function, number_intervals=10):
    assert number_intervals % 2 == 0
    interval_size = (b - a) / number_intervals
    I_cS2 = function(a) + function(b)
    for i in range(1, number_intervals, 2):
        I_cS2 += 4 * function(a + i * interval_size)
    for i in range(2, number_intervals-1, 2):
        I_cS2 += 2 * function(a + i * interval_size)
    return I_cS2 * (interval_size / 3.0)

def weddles_rule(a, b, function, number_intervals=10):
    S = simpsons_composite_rule(a, b, function, number_intervals)
    S2 = simpsons_composite_rule(a, b, function, number_intervals*2)
    return S2 + (S2 - S)/15.





def f(x):
    return x**2


x = np.linspace(0, 2, 10)
print(trapizoid_inti(f, x))
print(trap_comp_inti(f, x))
print(composite_sim_inti(f, 0.0, 2.0, 10))
