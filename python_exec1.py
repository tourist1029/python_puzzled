import sys
import math
import array
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import spline
from scipy.linalg import solve
import scipy
import scipy.linalg as sl


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
V = np.array([1.,2.])
N = np.array([[4.,5.],[7.,9.],[8., 6.]])
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
# A = {1,2,3,4,4}
# B = {5}
# C = A.union(B)
# D = A.intersection(C)
# E = C.difference(A)
# print(A)
# print(C)
# print(D)
# print(E)

# print('矩阵向量乘积为：', np.dot(V, M))
# print('矩阵与矩阵乘积：', np.dot(N, M))

# A = np.array([[1., 2.],[3., 4.]])
# b = np.array([1., 4.])
# x = solve(A, b)
# print(type(x))
# print(type(A))
# print(type(b))
# print(x)
# print(np.allclose(np.dot(A, x), b))
# print(A.shape)
# print(N.ndim) #表示列数，即维数  
# print(N)
# print(N.shape)
# print(N.dtype)
# print(N.strides)
# #创建复数数组
# # vv = np.array([1, 2, 5, 6, 7], dtype=complex)
# # print(vv)

# id = np.array([[1, 0],
#                 [0, 1]])

print('****************************************************')
# kk1 = np.zeros((3, 3))
# kk2 = np.ones((3, 3))
# kk3 = np.diag((3, 4, 5, 6))
# kk4 = np.random.rand(4,4)
# kk5 = np.arange(7, dtype=float)
# kk6 = np.linspace(1, 10 , 10)
# kk7 = np.zeros_like(N)
# print(kk1)
# print(kk2)
# print(kk3)
# print(kk4)
# print(kk5)
# print(kk6)
# print(kk7)

# MM = np.identity(5)
# print(MM)
# print(MM.shape)
# print(np.shape(MM))

# T = np.zeros((2, 5))
# print(T)
# print(T.ndim)

# vv = np.array([0,1,2,3,4,5])
# mm = vv.reshape(2,3)
# nn = mm.T
# print(vv)
# print(mm)
# print(vv)
# print(nn)
# kk = np.array([1,3,45,6,7,8])
# ll = kk.T
# print(kk)
# print(ll)

# AA = np.array([[1,2,3,4], [5,6,7,8]])
# print(AA)
# print(AA.sum(axis=0))
# print(AA.sum(axis=1))

MM = np.array([[1, 2],[3, 4]])
V = M[0,:]

V[-1] = 0
print(MM)
print(V)

print(V.base)
print(MM.base)

NN = np.array(MM.T)
print(NN)
print(NN.base is None)

A = np.array([[1,2],[3,4]])
B = np.array([[1,2],[3,4]])

print((A == B).all())




print('for the master branch')









