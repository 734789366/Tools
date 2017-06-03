# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:29:27 2017

@author: Administrator
"""
'''
loadtxt()读取了一个包含股价数据的CSV文件，用delimiter参数指定了文件中的分隔符为英文符号，
用usecols中的参数制定了我们感兴趣的数据列，并将UNpack参数设置为True使得不同的数据分开存储，
以便随后使用。
'''

import numpy as np
import datetime
import matplotlib.pyplot as plt

c, v = np.loadtxt('../Code/ch3code/data.csv', delimiter=',', usecols=(6, 7), unpack=True)

# 成交量加权平均价格VWAP
# 代表了金融资产的‘平均’价格，某个成交量的价格越高，该价格所占的比重越大
# VWAP是以成交量为权重计算出来的加权平均价格
weight = v/v.sum()
vmap = c.dot(weight)
print('VMAP: ', vmap)
print('VMAP:', np.average(c, weights=v))

print('mean:', c.sum()/len(c))
print('mean:', np.mean(c))

# 时间加权平均价格TWAP是另一种计算‘平均’价格的指标，其基本思想是，最近的价格重要性更大一些，
# 所以越近的价格权重越高。简单的方法是创建一个从0开始增长的自然数序列，自然数的个数即收盘价的个数
print('TWAP:', np.average(c, weights=np.arange(len(c))))

# 最大值，最小值，取值范围
print('c.max():', c.max())
print('np.max(c):', np.max(c))
print('c.min():', c.min())
print('np.min(c):', np.min(c))
print('c.ptp():', c.ptp())
print('np.ptp(c)', np.ptp(c))

# 在数据处理中，我们可以用一些阈值来去除异常值。
# 另一种好的方法就是中位数。
# 寻找中位数有两种方法，一种是调用函数np.median(),另一种是手工计算
print('median:', np.median(c))
sorted = np.msort(c)
if len(sorted)%2 == 0:
    median = (sorted[int(len(sorted)/2)] + sorted[int(len(sorted)/2-1)])/2
else:
    median = sorted[(len(c)-1)/2]
print('median:', median)

# 方差能够体现变量的变化程度，股价变动过于剧烈的股票会给持有者带来麻烦
# 计算方差有两种方法，一种是调用函数np.var(),另一种是根据定义手工计算，
# 所有数据距离算术平均值的差值的平方和的均值
print('np.var(c):', np.var(c))
print('var:', np.mean((c-np.mean(c))**2))

# 对股票收盘价的分析常常是基于股票收益率和对数收益率的。
# 简单收益率是指相邻两个价格之间的变化率,即差值除以前一日的收盘价
# 对数收益率是指所有价格取对数后，两两之间的差值。
# 投资者最感兴趣的实际是收益率的方差会标准差，因为这反映了投资风险的大小
# numpy的diff函数可以返回一个由相邻数组元素的差值构成的数组，不过diff返回的数组比收盘价数组少一个元素
print('c=\n', c)
print('np.diff(c)=\n', np.diff(c))
standard_returns = np.diff(c)/c[:-1]
log_returns = np.diff(np.log(c))
print('标准收益率：', standard_returns)
print('对数收益率：', log_returns)
print('标准收益率方差:', np.var(standard_returns))
print('标准收益率标准差:', np.std(standard_returns))
print('对数收益率方差:', np.var(log_returns))
print('对数收益率标准差:', np.std(log_returns))
print('标准收益率为正的索引:', np.where(standard_returns > 0))
print('对数收益率为正的索引:', np.where(log_returns > 0))

# 在投资学中，波动率volatility是对价格变动的一种度量，历史波动率可以根据历史价格数据计算得出
# 计算历史波动率时，需要用到对数收益率。
# 年波动率等于对数收益率的标准差除以其均值，再除以交易日倒数的平方根，通常取交易日为252天
# 注意： 在Python中，整数除法和浮点数除法的运算机制不同，必须使用浮点数才能得到正确的结果
# annual_volatility = np.std(log_returns)/np.mean(log_returns)/np.sqrt(1.0/252.0)
# monthly_volatility = np.std(log_returns)/np.mean(log_returns)/np.sqrt(1.0/12.0)

# 分析日期数据
# 将数据按照日期特征来进行处理
# 因为Numpy是面向浮点数运算的，而日期是字符串，因此需要对日期做一些专门的处理
# 通过定义一个专门的日期转换函数来进行日期转换
# 读取出来的日期是一个bytes，要将其转换为string，可以使用decode()，括号内可以加上编解码类型
# 星期一 0
# 星期二 1
# 星期三 2
# 星期四 3
# 星期五 4
def datestr2num(date):
    return datetime.datetime.strptime(date.decode('utf-8'), '%d-%m-%Y').date().weekday()

date, open, high, low, close = np.loadtxt('../Code/ch3code/data.csv', delimiter=',', usecols=(1, 3, 4, 5, 6), converters={1: datestr2num}, unpack=True)
print('date:', date)
print('date.shape', date.shape)
for i in range(5):
    indice = np.where(date == i)
    price = np.take(close, indice)
    print('date: %d, mean price: %f' % (i, np.mean(price)))

# 周汇总数据
# 因为有节假日等原因，所以数据并不是总包含完整的一周，所以只考虑了前三周的数据
close = close[:16]
dates = date[:16]
print(close.shape)
# 找到第一个星期一
first_monday = np.ravel(np.where(dates == 0))[0]
print(first_monday)
# 找到最后一个星期五
last_friday = np.ravel(np.where(dates == 4))[-1]
print(last_friday)
week_indice = np.arange(first_monday, last_friday+1)
week_indice = np.split(week_indice, 3)
#week_close = np.take(close, week_indice)
#week_close = np.split(week_close, 3)
#print(week_close)
# 在Numpy中，数组的维度也被称作轴，apply_along_axis函数会将函数作用于指定的轴上的每一个元素上
def summaries(indice, o, h, l, c):
    # 一周每天的盘后数据，包括开盘价，最高价，最低价，收盘价
    monday_open = o[indice[0]]
    week_high = np.max(np.take(h, indice))
    week_low = np.min(np.take(l, indice))
    friday_close = c[indice[-1]]
    return (monday_open, week_high, week_low, friday_close)
print('week_indice', week_indice)
week_summary = np.apply_along_axis(summaries, 1, week_indice, open, high, low, close)
print(week_summary)

# 真实波动幅度均值ATR是一个用来衡量股价波动性的技术指标
# ATR是基于N个交易日的最高价和最低价进行计算的，通常取最近20个交易日
# 对于每一个交易日，计算以下各项：
# h -l : 当日股价范围
# h - pre_close : 当日最高价和前一日收盘价之差
# pre_close - l : 前一日收盘价和当日最低价之差
# 真实波动幅度就是这三者的最大值
# maximum()函数比较多个数组的元素，返回对应位置上的最大值
date, open, high, low, close = np.loadtxt('../Code/ch3code/data.csv', delimiter=',', usecols=(1, 3, 4, 5, 6), converters={1: datestr2num}, unpack=True)
N = 20
h = high[-N:]
l = low[-N:]
pre_close = close[-N-1:-1]
true_range = np.maximum(h-l, h-pre_close, pre_close-l)
print(true_range)

atr = np.zeros(N)
atr[0] = np.mean(true_range)
for i in range(1, N):
    atr[i] = ((N-1)*atr[i-1]+true_range[i])
    atr[i] /= N
#atr /= N
np.set_printoptions(suppress=True)
print('atr:', atr)

# 简单移动平均线SMA通常用于分析时间序列上的数据。
# 为了计算它，我们需要定义一个N个周期的移动窗口，按照时间序列滑动这个窗口，
# 并计算窗口内的数据的均值。
# 通过Numpy的convolve卷积函数是个很好的选择
# 经过convolve的数据的大小为origin-N+1
N = 5
weights = np.ones(N)/N
sma = np.convolve(weights, close)[N-1:-N+1]
t = np.arange(N-1, len(close))
plt.plot(t, close[N-1:], 'r-')
plt.plot(t, sma, 'g-')
plt.show()

# 指数移动平均线EMA也是一种流行的技术指标，
# 指数移动平均线使用的权重是指数衰减的，
# 对历史上的数据点赋予的权重以指数速度减小
# 使用exp()和linspace()方法
N = 5
weights = np.exp(np.linspace(-1, 0, N))
weights /= np.sum(weights)
print(weights)
ema = np.convolve(weights, close)[N-1:-N+1]
t = np.arange(N-1, len(close))
plt.plot(t, close[N-1:], 'r-')
plt.plot(t, ema, 'g-')
plt.plot(t, sma, 'b-')
plt.show()

# 布林带又是一种技术指标，用以刻画价格波动区间。
# 布林带的基本形态是由三条轨道线组成的带状通道，上中下各一条
# 中轨，简单移动平均线
# 上轨，比简单移动平均线高两倍标准差距离，这里所说的标准差是指简单移动平均线的标准差
# 下轨，比简单移动平均线低2倍的标准差距离
deviation = []
c = len(close)
for i in range(0, c-N+1):
    dev = close[i:i+N]
    sma_std = np.std(dev)
    deviation.append(sma_std)
    print("i: sma_std:", (i,sma_std))
print(deviation)
upperBB = sma + 2*np.array(deviation)
lowerBB = sma - 2*np.array(deviation)
plt.plot(t, sma, 'r-')
plt.plot(t, upperBB, 'g-')
plt.plot(t, lowerBB, 'b-')
plt.show()

# Numpy中的linalg包是专门用于线性代数计算的，下面的计算基于一个假设，就是一个价格可以根据N个
# 之前的价格利用现行模型计算得出，也就是说，这个股价等于之前的股价与各自的系数相乘，在做加和的结果。
# 用线性代数的术语来讲，这就是一个最小二乘法的问题
def linear_predict():
    b = close[-N:]
    b = b[::-1]
#    print('b:', b)
    A = np.zeros([N,N])
    # 假设当前价格只跟前N个价格有关，则取前N个价格来填充矩阵A
    for i in range(N):
        A[i:] = close[-1-i-N:-1-i]
#        print('A:', A)
        # 系数向量，残差数组，A的秩，A的奇异值
        (x, residuals, rank, s) = np.linalg.lstsq(A, b)
 #       print(x, residuals, rank, s)
        # 得到了系数x，我们就可以预测下一次股价了
    print("N=", N, np.dot(x, b))
for N in range(5, 15):
    linear_predict()

# 趋势线描绘的是价格变化的趋势，是根据股价走势图上很多枢轴点绘成的曲线
# 枢轴点，假设等于最高价、最低价、收盘价的算术平均值
# 阻力位，指股价上升时遇到阻力，在转跌前的最高价格
# 支撑位，指在反弹前的最低价格
high, low, close = np.loadtxt('../Code/ch3code/data.csv', delimiter=',', usecols=(4, 5, 6), converters={1: datestr2num}, unpack=True)
pivots = (high+low+close)/3
print('pivots:', pivots.shape)

def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    return np.linalg.lstsq(A, y)[0]

t = np.arange(len(close))
print('t=', t.shape)
sa, sb = fit_line(t, pivots-(high-low))
ra, rb = fit_line(t, pivots+(high-low))
print('sa, sb', sa, sb)
print('ra, rb', ra, rb)
support = sa * t + sb
resistence = ra * t + rb
print('support', support)
print('resistence', resistence)
condition = (close < resistence) & (close > support)
print(close < resistence)
print(close > support)
print(condition)
print(len(np.where(condition)[0]))
print(support[condition])
print('ratio=', len(np.where(condition)[0])/len(close))

#除了np.where()之外，还有一种来计算支撑位和阻力位之间数据的个数的方法
a1 = close[close>support]
a2 = close[close<resistence]
print(a1.shape)
print(a2.shape)
print(np.intersect1d(a1, a2))
print(len(np.intersect1d(a1, a2))/len(close))
next_support = (t[-1]+1)*sa + sb
next_resistence = (t[-1]+1)*ra + rb
print('next_support', next_support)
print('next_resistence', next_resistence)
plt.plot(t, support, 'r-', label='support')
plt.plot(t, resistence, 'g-', label='resistence')
plt.plot(t, close, 'b-', label='close')
plt.legend()
plt.show()

# 数组的裁剪与压缩
# clip方法返回一个修正过的数组，将所有比给定最大值还打的元素设置为最大值，
# 将所有比给定最小值还小的元素设置为最小值
a = np.arange(6)
print(a)
print('clip:', np.clip(a, 2, 4))
# compress()方法返回一个根据给定条件筛选后的数组
print('compress:', a.compress(a>3))

# 阶乘
a = np.arange(1, 8)
print('prod', np.prod(a))
print('cumprod', np.cumprod(a))