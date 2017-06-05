#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:35:40 2017

@author: tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt

# linalg 模块
# fft 模块
# 随机数
# 连续分布和离散分布

# numpy.linalg模块包含线性代数的函数。使用这个模块,我们可以计算逆矩阵、
# 求特征值、解线性方程组以及求解行列式等
# 在线性代数中,矩阵A与其逆矩阵 相乘后会得到一个单位矩阵I,。
# numpy.linalg 模块中的 inv 函数可以计算逆矩阵.
A = np.mat('0 1 2; 1 0 3; 4 -3 8')
print 'np.linalg.inv(A):\n', np.linalg.inv(A)
print "A.I:\n", A.I

# 矩阵可以对向量进行线性变换,这对应于数学中的线性方程组。 numpy.linalg 中的函数
# solve可以求解形如Ax = b的线性方程组,其中A为矩阵,b为一维或二维的数组,x是未知变
# 量。我们将练习使用dot函数,用于计算两个浮点数数组的点积.
A = np.mat("1 -2 1; 0 2 -8; -4 5 9")
b = np.array([0, 8, 9])
print "Ax=b", np.linalg.solve(A, b)

# 特征值(eigenvalue)即方程Ax = ax的根,是一个标量。其中,A是一个二维矩阵,x是一个一维向量.
# 特征向量(eigenvector)是关于特征值的向量。在 numpy.linalg 模块中, eigvals
# 函数可以计算矩阵的特征值,而 eig 函数可以返回一个包含特征值和对应的特征向量的元组
A = np.mat('3 -2; 1 0')
print "特征值:", np.linalg.eigvals(A)
print "特征值 特征向量:", np.linalg.eig(A)

# SVD(Singular Value Decomposition,奇异值分解)是一种因子分解运算,将一个矩阵分解为3个矩阵的乘积。
# 奇异值分解是前面讨论过的特征值分解的一种推广。在 numpy.linalg 模块中的 svd 函数可以对矩阵进行奇异值分解
# 该函数返回3个矩阵——U、Sigma和V,其中U和V是正交矩阵,Sigma包含输入矩阵的奇异值.
A = np.mat('4 11 14; 8 7 -2')
U, sigma, V = np.linalg.svd(A, full_matrices=False)
print "U: ", U
print "sigma: ", sigma
print "V: ", V
print U*np.diag(sigma)*V

# 摩尔·彭罗斯广义逆矩阵(Moore-Penrose pseudoinverse)可以使用 numpy.linalg 模块中的pinv 函数进行求解
A = np.mat('4 11 14; 8 7 -2')
pseudoinv = np.linalg.pinv(A)
print "pseudoinv:", pseudoinv
print "A*pseudoinv", A*pseudoinv

# 行列式(determinant)是与方阵相关的一个标量值,对于一个n×n的实数矩阵,行列式描述的是一个线性变换
# 对“有向体积”所造成的影响。行列式的值为正表示保持了空间的定向(顺时针或逆时针),为负则表示颠倒了空间的定向。
# numpy.linalg 模块中的 det 函数可以计算矩阵的行列式.
A = np.mat("3 4; 5 6")
print "行列式: ", np.linalg.det(A)

# FFT(Fast Fourier Transform,快速傅里叶变换)是一种高效的计算DFT(Discrete Fourier Transform,离散傅里叶变换)的算法。
# FFT算法比根据定义直接计算更快,计算复杂度为O(NlogN) 。DFT在信号处理、图像处理、求解偏微分方程等方面都有应用。
# 在NumPy中,有一个名为 fft 的模块提供了快速傅里叶变换的功能。在这个模块中,许多函数都是成对存在的,也就是说许多函数存在
# 对应的逆操作函数。例如, fft 和 ifft 函数就是其中的一对.
x = np.linspace(0, 2*np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
itransformed = np.fft.ifft(transformed)
print np.all((wave-itransformed) < 10**-9)
plt.plot(transformed)
plt.show()
plt.plot(itransformed, 'g--')
plt.plot(wave, 'r*')
plt.show()

# numpy.linalg模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央,ifftshift 函数则是其逆操作
shifted = np.fft.fftshift(transformed)
plt.plot(shifted)
ishifted = np.fft.ifftshift(shifted)
print np.all((transformed-ishifted) < 10 ** -9)
plt.show()

# 随机数在蒙特卡罗方法(Monto Carlo method)、随机积分等很多方面都有应用。真随机数的产生很困难,因此在实际应用中
# 我们通常使用伪随机数。在大部分应用场景下,伪随机数已经足够随机,当然一些特殊应用除外。有关随机数的函数可以在NumPy的
# random模块中找到。随机数发生器的核心算法是基于马特赛特旋转演算法(Mersenne Twister algorithm)的。随机数可以从离散
# 分布或连续分布中产生。分布函数有一个可选的参数 size ,用于指定需要产生的随机数的数量。该参数允许设置为一个整数或元组,
# 生成的随机数将填满指定形状的数组。支持的离散分布包括几何分布、超几何分布和二项分布等.

# 二项分布是n个独立重复的是/非试验中成功次数的离散概率分布,这些概率是固定不变的,与试验结果无关.
# 对一个二项分布进行采样（size表示采样的次数），参数中的n, p分别对应于公式中的n,p，
# 函数的返回值表示n中成功（success）的次数（也即N）
print np.true_divide(np.sum(np.random.binomial(10, 0.5, 1000)),10*1000)
outcome = (np.random.binomial(9, 0.5, 10000))
cash = np.zeros(len(outcome)+1)
cash[0]=1000
for i in range(len(outcome)):
    if outcome[i] < 5:
        cash[i+1] = cash[i] -1
    else:
        cash[i+1] = cash[i] +1
print "max %d, min %d" % (cash.max(), cash.min())
plt.plot(cash)
plt.show()

# 超几何分布(hypergeometric distribution)是一种离散概率分布,它描述的是一个罐子里有两种物件,
# 无放回地从中抽取指定数量的物件后,抽出指定种类物件的数量。NumPy random模块中的hypergeometric函数可以模拟这种分布
# np.random.hypergeometric(good, bad, sample_count, sample_times)
N = 1000
points = np.zeros(N+1)
outcome = np.random.hypergeometric(25, 1, 3, N)
for i in range(N):
    if outcome[i] == 3:
        points[i+1] = points[i]+1
    else:
        points[i+1] = points[i]-6
plt.plot(points)
plt.show()

# 连续分布可以用PDF(Probability Density Function,概率密度函数)来描述。
# 随机变量落在某一区间内的概率等于概率密度函数在该区间的曲线下方的面积。

# 随机数可以从正态分布中产生,它们的直方图能够直观地刻画正态分布.使用NumPy random模块中
# 的normal函数产生指定数量的随机数.
# returns:
#   n: the values of the histogram bins
#   bins: the edges of the bins
#   list used to create the histogram
N = 10000
mu = 1.0
sigma = 2.0
norm = np.random.normal(loc=mu, scale=sigma, size=N)
n, bins, patches = plt.hist(norm, int(np.sqrt(N)), normed=True, lw=1)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)))
plt.plot(bins[:-1], n, 'r--')
plt.show()

# 对数正态分布(lognormal distribution)是自然对数服从正态分布的任意随机变量的概率分布。NumPy random模块
# 中的lognormal函数模拟了这个分布
N = 10000
mu = 0
sigma = 1
log_norm = np.random.lognormal(mean=mu, sigma=sigma, size=N)
n, bins, patches = plt.hist(log_norm, int(np.sqrt(N)), normed=True)
x = np.linspace(min(bins), max(bins), len(bins))
pdf = np.exp(-(np.log(x)-mu)**2/(2*sigma**2))/(x*sigma*np.sqrt(2*np.pi))
plt.plot(x, pdf)
plt.show()