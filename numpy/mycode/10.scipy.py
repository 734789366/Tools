#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:23:00 2017

@author: tensorflow
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io, stats, signal, fftpack, optimize, integrate, interpolate, ndimage
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter
from PIL import Image

# MATLAB以及其开源替代品Octave都是流行的数学工具。scipy.io包的函数可以在Python中加载
# 或保存MATLAB和Octave的矩阵和数组。loadmat函数可以加载.mat文件。savemat函数可以将数组
# 和指定的变量名字典保存为.mat文件
a = np.arange(9)
io.savemat('iosavemat.mat', {'array': a})
b = io.loadmat('iosavemat.mat')
print 'b[array]:', b['array']

# SciPy的统计模块是 scipy.stats ,其中有一个类是连续分布的实现,一个类是离散分布的
# 实现。此外,该模块中还有很多用于统计检验的函数
# 使用 scipy.stats 包按正态分布生成随机数
generated = stats.norm.rvs(size=900)
# 用正态分布去拟合生成的数据,得到其均值和标准差
print stats.norm.fit(generated)
# 偏度(skewness)描述的是概率分布的偏斜(非对称)程度.偏度检验有两个返回值,其中第二个
# 返回值为p-value,即观察到的数据集服从正态分布的概率
print 'skewness:', stats.skewtest(generated)
# 峰度(kurtosis)描述的是概率分布曲线的陡峭程度,峰度检验与偏度检验类似,当然这里是针对峰度
print 'Kurtosistest', stats.kurtosistest(generated)
# 正态性检验(normality test)可以检查数据集服从正态分布的程度,该检验同样有两个返回值,
# 其中第二个返回值为p-value
print "Normaltest", stats.normaltest(generated)
# 使用SciPy我们可以很方便地得到数据所在的区段中某一百分比处的数值
print "95 percentile", stats.scoreatpercentile(generated, 95)
# 也可以从数值1出发找到对应的百分比
print "Percentile at 1", stats.percentileofscore(generated, 1)

plt.hist(generated)
plt.show()

# scipy.signal 模块中包含滤波函数和B样条插值(B-spline interpolation)函数.
# SciPy中以一组数值来定义信号。我们以detrend函数作为滤波器的一个例子。该函数可以
# 对信号进行线性拟合,然后从原始输入数据中去除这个线性趋势
close = np.loadtxt("../Code/ch7code/AAPL.csv", delimiter=',', usecols=(6), unpack=True)
date = np.loadtxt("../Code/ch7code/AAPL.csv", delimiter=',', usecols=(1), unpack=True, dtype='str')
y = signal.detrend(close)
alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter('%b %Y')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(close, 'o')
plt.plot(close-y, '-')
plt.show()

# 现实世界中的信号往往具有周期性。傅里叶变换(Fourier transform)是处理这些信号的常用工具。
# 傅里叶变换是一种从时域到频域的变换,也就是将周期信号线性分解为不同频率的正弦和余弦函数
# 傅里叶变换的函数可以在scipy.fftpack模块中找到(NumPy也有自己的傅里叶工具包,即numpy.fft)
# 这个模块包含快速傅里叶变换、微分算子和拟微分算子以及一些辅助函数.
# 应用傅里叶变换,得到信号的频谱
amps = np.abs(fftpack.fftshift(fftpack.rfft(y)))
amps[amps < 0.1*amps.max()] = 0
plt.plot(y, 'o', label='detrended')
plt.plot(amps, '^', label='amps')
plt.plot(-fftpack.irfft(fftpack.ifftshift(amps)), label='filtered')
plt.legend()

# 优化算法(optimization algorithm)尝试寻求某一问题的最优解,例如找到函数的最大值或最小值,
# 函数可以是线性或者非线性的。解可能有一些特定的约束,例如不允许有负数。在scipy.optimize模块
# 中提供了一些优化算法,最小二乘法函数leastsq就是其中之一.当调用这个函数时,我们需要提供一个
# 残差(误差项)函数。这样, leastsq将最小化残差的平方和.
def residuals(p, y, x):
    A, k, theta, b = p
    err = y - A*np.sin(2*np.pi*k*x + theta) + b
    return err

filtered = -fftpack.irfft(fftpack.ifftshift(amps))
N = len(close)
f = np.linspace(-N/2, N/2, N)
p0 = [filtered.max(), f[amps.argmax()]/(2*N), 0, 0]
print 'p0: ', p0
plt.show()

# SciPy中有数值积分的包scipy.integrate,在NumPy中没有相同功能的包。quad函数可以求
# 单变量函数在两点之间的积分,这些点之间的距离可以是无穷小或无穷大。该函数使用最简单
# 的数值积分方法即梯形法则(trapezoid rule)进行计算

# 高斯积分(Gaussian integral)出现在误差函数(数学中记为erf)的定义中,但高斯积分本身的
# 积分区间是无穷的,它的值等于pi的平方根。我们将使用quad函数计算它
print "Gaussian integral", np.sqrt(np.pi), integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)

# 插值(interpolation)即在数据集已知数据点之间“填补空白”。scipy.interpolate函数可以根据
# 实验数据进行插值。interp1d类可以创建线性插值(linear interpolation)或
# 三次插值(cubic interpolation)的函数。默认将创建线性插值函数,三次插值函数可以通过设置kind参数
# 来创建。interp2d类的工作方式相同,只不过用于二维插值.
# 我们将使用sinc函数创建数据点并添加一些随机噪音,我们将进行线性插值和三次插值,并绘制结果
x = np.linspace(-18, 18, 36)
noise = 0.1 * np.random.random(len(x))
signal = np.sinc(x) + noise

interpreted = interpolate.interp1d(x, signal)
x2 = np.linspace(-18, 18, 180)
y = interpreted(x2)

cubic = interpolate.interp1d(x, signal, kind='cubic')
y2 = cubic(x2)
plt.plot(x, signal, 'o', label='data')
plt.plot(x2, y, '-', label='linear')
plt.plot(x2, y2, '-', lw=2, label='cubic')
plt.legend()
plt.show()

# 我们可以使用scipy.ndimage包进行图像处理.该模块包含各种图像滤波器和工具函数.
# 在 scipy.misc 模块中,有一个函数可以载入Lena图像
image = Image.open('lena.png', 'r')
print image
plt.subplot(221)
plt.title('Original Image')
plt.imshow(image, cmap=plt.cm.gray)
