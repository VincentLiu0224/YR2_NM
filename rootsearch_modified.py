import numpy as np
import math as mt


def main(f, x1, x2):
    result_list = []
    b = x2
    while x1 <= b:
        x1, x2 = indi(f, x1, x2)
        try:
            diff = x1 - result_list[-1]
        except IndexError:
            inplant(f, x1, x2, result_list, b)
        except TypeError:
            break
        else:
            test = bool(diff >= 1e-4)
            if test is True:
                inplant(f, x1, x2, result_list, b)
        x1 = x2
        x2 = b
    return result_list


def inplant(f, x1, x2, result_list, b):
    if abs(f(x1)) < 1e-4:
        result_list.append(x1)
    elif abs(f(x2)) < 1e-4:
        result_list.append(x2)
    else:
        if x2 - x1 <= 1e-6:
            result = x2
            result_list.append(result)
        elif x2 - x1 > 1e-6:
            result = bisection(f, x1, x2)
            if result is not None and result <= b:
                result_list.append(result)
        else:
            result_list.append("fail")
            result = "fail"
    return


def indi(f, x1, x2):
    for i in range(10):
        dx = 0.05
        x1, x2 = rootsearch(f, x1, x2, dx)
        if abs(f(x1)) < 1e-2 or abs(f(x2)) < 1e-2:
            break
        else:
            continue
    return x1, x2


def rootsearch(f, x1, x2, dx):
    x3 = x1 + dx
    f1 = f(x1)
    f3 = f(x3)
    while np.sign(f1) == np.sign(f3):
        if x1 >= x2:
            break
        elif x3 - x1 <= 1e-2:
            break
        else:
            x1 = x1 + dx
            x3 = x1 + dx
            f1 = f(x1)
            f3 = f(x3)
            if f1 - 0.0 < 1e-6 or f3 - 0.0 < 1e-6:
                break
            else:
                continue
    return x1, x3


def bisection(f, x1, x2):
    f1 = f(x1)
    f2 = f(x2)
    if abs(f1) < 1e-9:
        return x1
    elif abs(f2) < 1e-9:
        return x2
    else:
        if np.sign(f1) == np.sign(f2):
            return None
        else:
            n = int(np.ceil(mt.log(abs(x2 - x1)/1e-9)/mt.log(2.0)))
            for i in range(n):
                x4 = 0.5*(x1 + x2)
                f4 = f(x4)
                if (abs(f4) > abs(f1)) and (abs(f4) > abs(f2)):
                    return
                elif abs(int(f4) - 0.0) < 1e-6:
                    return x4
                elif np.sign(f2) != np.sign(f4):
                    x1 = x4
                    f1 = f4
                else:
                    x2 = x4
                    f2 = f4
    return (x1 + x2)/2.0
