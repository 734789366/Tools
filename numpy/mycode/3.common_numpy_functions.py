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
