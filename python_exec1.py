import sys
import math
import array
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

#\t,\n的用法
print('Temperature:\t20\tC\nPressure:\t5\tPa.')
#格式化字符串用法
quantity = 33.4785
print("{name} {value:3.2f}".format(name="quantity", value=quantity))
print("we {} in LaTex \\begin {{equation}}".format('like'))

#列表合并zip， 按照长度最小的列表合并，返回一个元组列表
ind = [0,1,2,3,4]
color = ['red', 'green', 'black', 'blue']
kk = list(zip(color,ind))
print(kk)

L = [2,3,10,1,5]
L2 = [x*2 for x in L]
L3 = [x*2 for x in L if 4<x<=10]
print(L2)
print(L3)
# M = [[1,2,3],[4,5,6]]
# flat = [M[i][j] for i in range(2) for j in range(3)]
# print(M)
# print(flat)

print("for the test reset")

M = np.array([[1.,2.],[3.,4.]])
V = np.array([1.,2.,3.])
print(V[0])
print(V[:2])
print(M[0,1])
V[:2] = [10, 20]
print(V[:2])
print(V)
print("M lengths is ", len(M))
print('V length is: ', len(V))

print("for the f1 branch!")
print('for the second f1 commit')

#集合用{}表示，并自动取消重复的元素
A = {1,2,3,4,4}
B = {5}
C = A.union(B)
D = A.intersection(C)
E = C.difference(A)
print(A)
print(C)
print(D)
print(E)

#线性代数部分内容
v = np.array([1,2,3,4,5])
print(type(v))

v1 = np.array([1.,2.,3.])
v2 = np.array([2, 0, 1.])
print(v1)
print(v2)
print(2*v1)
print(v1/2)
print(3*v1)
print(3*v1+2*v2)





print('for the f1 branch')









