# course's methods of newton-raphson, quasi nr and scant method
# suscepytable to numerical overflow, sensible to start point
def newton(f, x0, dfdx, atol=1.0e-6):
    x = [x0]
    fevals = 0
    while True:
        x.append(x[-1] - f(x[-1])/dfdx(x[-1]))
        fevals += 2
        if abs(x[-1]-x[-2]) < atol:
            print('Newton (analytical derivative) used', fevals, 'function evaluations')
            return x[-1]

def quasi_newton(f, x0, dx=1.0E-7, atol=1.0E-6):
    x = [x0]
    while True:
        dfdx = (f(x[-1] + dx) - f(x[-1]))/(dx)
        x.append(x[-1] - f(x[-1])/dfdx)
        if abs(x[-1]-x[-2]) < atol:
            return x[-1]

def secant(f, x0, x1, atol=1.0E-6):
    x = [x0, x1]
    while True:
        dfdx = (f(x[-1])-f(x[-2])) / (x[-1]-x[-2])
        x.append(x[-1] - f(x[-1])/dfdx)
        if abs(x[-1]-x[-2]) < atol:
            return x[-1]


def quasi_newton_for_backeu(f, x0, t0, dt, diff, dx=1e-7, atol=1.0E-6):
    x = [x0]
    while True:
        dfdx = (f(x[-1] + dx, x[0], t0+dt, dt, diff) - f(x[-1], x[0], t0+dt, dt, diff))/(dx)
        x.append(x[-1] - (f(x[-1], x[0], t0+dt, dt, diff)/dfdx))
        if abs(sum(x[-1]-x[-2])) < atol:
            return x[-1]
