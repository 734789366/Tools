#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:55:45 2017

@author: tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator
from matplotlib.finance import candlestick_ohlc
from datetime import date

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 绘制多项式函数的图像.使用NumPy的多项式函数poly1d 来创建多项式.
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
plt.plot(x, y, label="y=x**3+2*x**2+3*x+4")
plt.xlabel("x")
plt.ylabel('y(x)')
plt.legend()
plt.show()

# 绘制一个多项式函数,以及使用derive函数和参数m为1得到的其一阶导函数
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func1 = func.deriv(m=1)
x = np.linspace(-10, 10, 30)
y = func(x)
y1 = func1(x)
plt.plot(x, y, 'ro', x, y1, 'g--')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 绘图时可能会遇到图中有太多曲线的情况,而你希望分组绘制它们。这可以使用subplot函数完成
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
func1 = func.deriv(m=1)
y1 = func1(x)
func2 = func.deriv(m=2)
y2 = func2(x)

plt.subplot(311)
plt.plot(x, y, 'r-')
plt.title('Polynomial')

plt.subplot(312)
plt.plot(x, y1, 'b^')
plt.title('First Derivative')

plt.subplot(313)
plt.plot(x, y2, 'go')
plt.title('Second Derivative')
plt.show()

today = date.today()
start = (today.year -1, today.month, today.day)

# 我们需要创建所谓的定位器(locator),这些来自matplotlib.dates包中的对象可以在x轴上定位月份和日期
alldays = DayLocator()
months = MonthLocator()
# 创建一个日期格式化器(date formatter)以格式化x轴上的日期。该格式化器将创建一个字符串,包含简写的月份和年份
month_formatter = DateFormatter("%b %Y")
data = np.loadtxt("../Code/ch7code/AAPL.csv", delimiter=',', usecols=(3, 4, 5, 6), unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)

candlestick_ohlc(ax, data)
fig.autofmt_xdate()
plt.show()

close = np.array(data[3])
print "len(close)", len(close)
plt.hist(close)
plt.show()

# 三维绘图非常壮观华丽,对于3D作图,我们需要一个和三维投影相关的Axes3D对象.
# z = x**2 + y**2
fig = plt.figure()
# 使用 3d 关键字来指定图像的三维投影
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')
u = np.linspace(-1, 1, 100)
# 使用 meshgrid 函数创建一个二维的坐标网格
x, y = np.meshgrid(u, u)
z = x**2 + y**2
# 指定行和列的步幅,以及绘制曲面所用的色彩表(color map)。步幅决定曲面上“瓦片”的大小,而色彩表的选择取决于个人喜好
ax1.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r)

# Matplotlib中的等高线3D绘图有两种风格——填充的和非填充的。我们可以使用contour函数创建一般的等高线图。
# 对于色彩填充的等高线图,可以使用contourf绘制.
ax2.contourf(x, y, z)
plt.show()

# Matplotlib提供酷炫的动画功能。Matplotlib中有专门的动画模块。我们需要定义一个回调函数,用于定期更新屏幕上的内容。
# 我们还需要一个函数来生成图中的数据点.
# 绘制三个随机生成的数据集,分别用圆形、小圆点和三角形来显示。不过,我们将只用随机值更新其中的两个数据集
fig = plt.figure()
ax = fig.add_subplot(111)
N = 20
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
ax.set_ylim(0, 1)
circles, triangles, dots = ax.plot(x, 'ro', y, 'g^', z, 'b.')

# 下面的函数将被定期调用以更新屏幕上的内容
# generate生成随机数据，update用生成的随机数据来更新画面
def update(data):
    circles.set_ydata(data[0])
    triangles.set_ydata(data[1])
    dots.set_ydata(data[2])
    return circles, triangles

def generate():
    while True: yield np.random.rand(3, N)

anim = animation.FuncAnimation(fig, update, generate, interval=500)
plt.show()
