import numpy as np


def main_bracketing(f, a, b):
    for i in range(5):
        dx = (b - a)/10
        a, b = bracketing(f, a, b, dx)
    return (a + b)/2


def main_bisection(f, x1, x2):
    result_list = []
    b = x2
    while x1 <= b:
        x1, x2 = indi(f, x1, x2)
        try:
            diff = abs(x1 - result_list[-1])
        except IndexError:
            result = inplant(f, x1, x2, result_list, b)
            result_list.append(result)
        except TypeError:
            print("no root in this bracket")
        else:
            if x1 >= b:
                break
            else:
                if diff >= 1e-3:
                    result = inplant(f, x1, x2, result_list, b)
                    if abs(result-result_list[-1]) >= 1e-2:
                        result_list.append(result)
                x1 = x2
                x2 = b
    return result_list


def indi(f, x1, x2):
    dx = 0.05
    x1, x2 = bracketing(f, x1, x2, dx)
    return x1, x2


def inplant(f, x1, x2, result_list, b):
    if abs(f(x1)) < 1e-4:
        return x1
    elif abs(f(x2)) < 1e-4:
        return x2
    else:
        if x2 - x1 <= 1e-6:
            result = x2
            return result
        elif x2 - x1 > 1e-6:
            result = bisection(f, x1, x2)
            if result is not None and result <= b:
                return result
        else:
            result = "fail"
    return result


def bracketing(f, a, b, dx, tol=1e-6):
    c = a + dx
    f1 = f(a)
    f2 = f(c)
    while np.sign(f1) == np.sign(f2):
        if a >= b:
            break
        else:
            a += dx
            c += dx
            f1 = f(a)
            f2 = f(c)
            if abs(f1) - 0.0 < tol or abs(f2) - 0.0 < tol:
                break
            else:
                continue
    return a, c


def bisection(f, a, b, tol=1e-6):
    c = (a+b)/2
    fa = f(a)
    fc = f(c)
    while fc >= 1e-6:
        if np.sign(fc) == np.sign(fa):
            a = c
        elif a >= b:
            return
        else:
            b = c
        c = (a+b)/2
        fc = f(c)
    return c
