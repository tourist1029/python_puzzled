import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import spline
import scipy


print("hello python math")
print(np.exp(1000.))
a = np.inf
print(3-a)
print(3+a)
print(a+a)
print(a-a)
print(a/a)
print(sys.float_info.epsilon)

a = math.pi
a1 = np.float64(a)
a2 = np.float32(a)
print(a-a1)
print(a-a2)

f32 = np.finfo(np.float32)
print("float32 is ", f32)
print(f32.precision)
f64 = np.finfo(np.float64)
print("float64 is ", f64)
print(f64.precision)
f = np.finfo(float)
print(f.precision)
print("*************************************")
print(f64.max)
print(f32.max)
print(f.max)

z = -5 + 2j
print(z)
print(z.real)
print(z.imag)






