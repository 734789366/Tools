# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:35:17 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 协方差描述的是两个变量共同变化的趋势，其实就是归一化前的相关系数。
# 使用cov函数计算股票收益率的协方差矩阵

bhp_close = np.loadtxt('../Code/ch4code/ch4code/BHP.csv', delimiter=',', usecols={6}, unpack=True)
vale_close = np.loadtxt('../Code/ch4code/ch4code/VALE.csv', delimiter=',', usecols={6}, unpack=True)
bhp_returns = np.diff(bhp_close)/bhp_close[:-1]
vale_returns = np.diff(vale_close)/vale_close[:-1]
#print(bhp_close)
#print(vale_close)
print('BHP收益率:\n', bhp_returns)
print('VALE收益率:\n', vale_returns)
covariance = np.cov(vale_returns, bhp_returns)
print('BHP, VALE收益率协方差矩阵：\n', covariance)
print('BHP, VALE收益率协方差矩阵对角线元素：\n', np.diagonal(covariance))
print('协方差矩阵的迹（对角线元素之和）:', np.trace(covariance))

# 两个向量的相关系数被定义为协方差除以各自标准差的内积
corr = covariance/(np.std(vale_returns)*np.std(bhp_returns))
print('相关系数矩阵：', corr)
# 相关系数被用来度量这两只股票的相关程度，数值的取值范围在-1到1之间。
# 根据定义，一组数值与自身的相关系数等于1，这是严格线性关系的理想值。
# np.corrcoef()函数可以被用于计算相关系数矩阵
corrcoef = np.corrcoef(bhp_returns, vale_returns)
# 对角线上的元素即BHP和VALE与自身的相关系数，均为1
print('相关系数矩阵:', corrcoef)
# 如果两只股票的差值偏离了平均差值2倍于标准差的距离，则认为这两只股票走势不同步
difference = bhp_close - vale_close
mean_difference = np.mean(difference)
std_difference = np.std(difference)
print('股价不同步：', np.abs(difference-mean_difference) > 2*std_difference)
t = np.arange(len(vale_returns))
plt.plot(t, vale_returns, 'r-', label='VALE close')
plt.plot(t, bhp_returns, 'g-', label='BHP close')
plt.show()

# NumPy中的polyfit函数可以用多项式取拟合一系列数据点，无论这些数据点是否来自连续函数
# 我们用一个三次多项式取拟合BHP和VALE两支股票收盘价的差值
# polyval()用于根据系数推断下一个值
N = 6
bhp_close = np.loadtxt('../Code/ch4code/ch4code/BHP.csv', delimiter=',', usecols={6}, unpack=True)
vale_close = np.loadtxt('../Code/ch4code/ch4code/VALE.csv', delimiter=',', usecols={6}, unpack=True)
difference = bhp_close - vale_close
t = np.arange(len(difference))
poly = np.polyfit(t, difference, deg=N)
print(poly)
print('预估下一个值:', np.polyval(poly, t[-1]+1))
# 理想情况下，BHP和VALE股价的差值越小越好
# np.roots()可用于计算拟合的多项式函数什么时候到达0值
print('多项式0点：', np.roots(poly))

# 极值，可能是函数的最大值或最小值
# 极值位于函数导数为0的位置，使用polyder函数对多项式函数求导
der = np.polyder(poly)
print('多项式的导函数：', der)
print('导函数的0点，也即原函数的极值点：', np.roots(der))
predict = np.polyval(poly, t)
print('多项式的预估值：\n',predict)
print('最大值：', np.argmax(np.polyval(poly, t)))
print('最小值：', np.argmin(np.polyval(poly, t)))
plt.plot(t, difference, label='真实值')
plt.plot(t, predict, label='预估值')
plt.legend()
plt.show()

# 净额成交量或能量潮指标OBV是简单的股价指标之一，可以表示价格波动的大小。
# OBV可以由当日收盘价、前一天的收盘价、以及当日成交量计算得出。
# 我们以前一日为基期计算当日的OBV值，若当日收盘价高于前一日收盘价，则本日OBV等于基期OBV加上当日
# 成交量。若当日收盘价低于前一日收盘价，则本日OBV等于基期OBV减去当日成交量
c, v = np.loadtxt('../Code/ch4code/ch4code/BHP.csv', delimiter=',', usecols={6, 7}, unpack=True)
#sign = np.sign(np.diff(c))
sign = np.piecewise(np.diff(c), [np.diff(c)<0, np.diff(c)>0], [-1, 1])
obv = v[1:]*sign
t = np.arange(len(obv))
plt.plot(t, obv)
print('obv', (obv))
plt.show()

# 模拟股票交易
# 使用vectorize函数可以避免在程序中使用循环
o, h, l, c = np.loadtxt('../Code/ch4code/ch4code/BHP.csv', delimiter=',', usecols={3, 4, 5, 6}, unpack=True)
# 该函数尝试以稍微低于开盘价的价格买入股票，如果这个价格不在当日的价格范围内，则买入失败，没有盈利也没有亏损。
# 否则，我们以收盘价卖出，所得利润即买入和卖出的差价。我们计算相对利润。
percent = 0.002
def calc_profit(open, high, low, close):
    buy = open * (1-percent)
    if low < buy < high:
        return (close - buy)/buy
    else:
        return 0

func = np.vectorize(calc_profit)
profits = func(o, h, l, c)
print('profits:', len(profits))
print('Not zero:', len(profits[profits!=0]))

print('盈利日期统计')
winning_trades = profits[profits>0]
print('盈利日期数', len(winning_trades))
print('盈利均值', np.mean(winning_trades))
losing_trades = profits[profits<0]
print('亏损日期数', len(losing_trades))
print('亏损均值', np.mean(losing_trades))

# 数据中包含噪声时，很难进行处理，因此我们通常需要对其进行平滑处理
# Numpy中提供了一个hanning函数来进行平滑处理，hanning是一个加权余弦的窗函数
bhp_close = np.loadtxt('../Code/ch4code/ch4code/BHP.csv', delimiter=',', usecols={6}, unpack=True)
vale_close = np.loadtxt('../Code/ch4code/ch4code/VALE.csv', delimiter=',', usecols={6}, unpack=True)
N = 8
weights = np.hanning(N)
weights /= weights.sum()
print('hanning weights', weights)
bhp_returns = np.diff(bhp_close)/bhp_close[:-1]
vale_returns = np.diff(vale_close)/vale_close[:-1]
smooth_bhp = np.convolve(weights, bhp_returns)[N-1:-N+1]
smooth_vale = np.convolve(weights, vale_returns)[N-1:-N+1]
t = np.arange(N-1, len(bhp_returns))
plt.plot(t, bhp_returns[N-1:], 'b-', lw=1.0)
plt.plot(t, smooth_bhp, 'b-', lw=2.0)
plt.plot(t, vale_returns[N-1:], 'r-', lw=1.0)
plt.plot(t, smooth_vale, 'r-', lw=2.0)
plt.show()
K=8
poly_bhp = np.polyfit(t, smooth_bhp, deg=K)
poly_vale = np.polyfit(t, smooth_vale, deg=K)
poly_sub = np.polysub(poly_bhp, poly_vale)
xpoints = np.select([np.isreal(np.roots(poly_sub))], [np.roots(poly_sub)])
xpoints = np.trim_zeros(xpoints)
print("实数交叉点：", np.real(xpoints))
