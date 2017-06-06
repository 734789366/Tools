#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:35:09 2017

@author: tensorflow
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt

# 排序和搜索
# 特殊函数
# 金融函数
# 窗口函数

# NumPy提供了多种排序函数,如下所示:
# sort函数返回排序后的数组;
# lexsort函数根据键值的字典序进行排序;
# argsort函数返回输入数组排序后的下标;
# ndarray类的sort方法可对数组进行原地排序;
# msort函数沿着第一个轴排序;
# sort_complex函数对复数按照先实部后虚部的顺序进行排序

# 使用NumPy中的lexsort函数对AAPL的收盘价数据进行了排序
def date2str(date):
    return datetime.datetime.strptime(date.decode('utf-8'), "%d-%m-%Y").toordinal()

dates, close = np.loadtxt("../Code/ch7code/AAPL.csv", delimiter=',', usecols=(1, 6), converters={1: date2str}, unpack=True)

print "close:", close
print "dates:", dates
indices = np.lexsort((dates, close))
print "indice:", indices
print ["%s %s"%(datetime.date.fromordinal(int(dates[i])), close[i]) for i in indices]

# NumPy中有专门的复数类型,使用两个浮点数来表示复数。这些复数可以使用NumPy的sort_complex函数进行排序.
# 该函数按照先实部后虚部的顺序排序.
complex_number = np.random.random(5) + 1j*np.random.random(5)
print "complex_number:", complex_number
print "Sorted\n", np.sort_complex(complex_number)

a = np.array([2, 1, 4, 8, 6, 9, 3])
print "np.sort(a): ", np.sort(a)
print "np.msort(a): ", np.msort(a)
print "np.argsort: ", np.argsort(a)
a.sort()
print "a :", a

# NumPy中有多个函数可以在数组中进行搜索:
# max返回数组中最大值,argmax函数返回数组中最大值对应的下标
# nanargmax函数提供相同的功能,但忽略NaN值
# min返回数组中最小值,argmin函数返回数组中最小值对应的下标
# nanargmin函数提供相同的功能,但忽略NaN值
# argwhere函数根据条件搜索非零的元素,并分组返回对应的下标
# searchsorted 函数可以为指定的插入值寻找维持数组排序的索引位置。
# 该函数使用二分搜索算法,计算复杂度为O(log(n))
# extract 函数返回满足指定条件的数组元素

# searchsorted 函数为指定的插入值返回一个在有序数组中的索引位置,从这个位置插入可以保持数组的有序性
a = np.arange(5)
indices = np.searchsorted(a, [-2, 7])
new_a = np.insert(a, indices, [-2, 7])
print "indice:", indices
print "new_a:", new_a

# 使用 extract 函数根据一个指定的布尔条件从数组中抽取了偶数元素
a = np.arange(9)
condition = (a%2==0)
print "偶数元素:", np.extract(condition, a)

# 窗函数(window function)是信号处理领域常用的数学函数,相关应用包括谱分析和滤波器设计等.
# 这些窗函数除在给定区间之外取值均为0. NumPy中有很多窗函数,如bartlett、blackman 、hamming 、hanning和kaiser
# 巴特利特窗(Bartlett window)是一种三角形平滑窗.调用NumPy中的bartlett函数,以计算巴特利特窗.
window = np.bartlett(42)
plt.plot(window)
plt.show()

# 布莱克曼窗(Blackman window)形式上为三项余弦值的加和. NumPy中的blackman函数返回布莱克曼窗。
# 该函数唯一的参数为输出点的数量。如果数量为0或小于0,则返回一个空数组
N = 30
close = np.loadtxt("../Code/ch7code/AAPL.csv", delimiter=',', usecols=(6), unpack=True)
print "close:", close
window = np.blackman(N)
smoothed = np.convolve(window/window.sum(), close, mode='same')
plt.plot(smoothed[N:-N], label='smoothed')
#plt.plot(close[N:-N], label='closes')

# 调用 hamming 函数,以计算汉明窗
window = np.hamming(42)
plt.plot(window)
plt.show()

# 凯泽窗(Kaiser window)是以贝塞尔函数(Bessel function)定义的
# 调用 kaiser 函数,以计算凯泽窗
window = np.kaiser(42, 14)
plt.plot(window)
plt.show()

# 贝塞尔函数( Bessel function )是贝塞尔微分方程的标准解函数.在NumPy中,以i0表示第一类修正的零阶贝塞尔函数.
# 使用 NumPy 的 linspace 函数生成一组均匀分布的数值
# 调用 i0 函数进行计算
# 使用 Matplotlib 绘制修正的贝塞尔函数
x = np.linspace(0, 4, 100)
vals = np.i0(x)
plt.plot(x, vals)
plt.show()

# sinc函数在数学和信号处理领域被广泛应用。NumPy中有同名函数sinc,并且也存在一个二维版本
x = np.linspace(0, 4, 100)
vals = np.sinc(x)
plt.plot(x, vals)
plt.show()

xx = np.outer(x, x)
vals = np.sinc(xx)
plt.plot(vals)
plt.imshow(vals)