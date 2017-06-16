#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:26:21 2017

@author: tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import mpl_toolkits.mplot3d.axes3d as p3

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
首先说明一下这里的三个变量分别是k（x轴）、b（y轴）以及ErrorArray（z轴）。
为了更好地理解mgrid后的k、b以及ErrorArray是什么，我想在这里举个简单的例子，
然后用Python做个图，这样大家就都明白了。
'''

'''
假设f(k,b)=3k^2+2b+1，k轴范围为1~3，b轴范围为4~6：

【step1：k扩展】(朝右扩展)：
[1 1 1]
[2 2 2]
[3 3 3]

【step2：b扩展】(朝下扩展)：
[4 5 6]
[4 5 6]
[4 5 6]

【step3：定位（ki，bi）】（把上面的k、b联合起来）：
[(1,4) (1,5) (1,6)]
[(2,4) (2,5) (2,6)]
[(3,4) (3,5) (3,6)]

【step4：将（ki，bi）代入f(k,b)=3k^2+2b+1求f(ki,bi)】
[12 14 16]
[21 23 25]
[36 38 40]
'''
k, b = np.mgrid[1:3:10j, 4:6:10j]
f_kb = 3*k**2 + 2*b + 1

#print k.shape, k
#print b.shape, b
#print f_kb.shape, f_kb
#
## 统统转成9行1列
#k.shape=-1, 1
#b.shape=-1, 1
#f_kb.shape=-1, 1
#
#fig, ax = plt.subplots()
#ax3d = p3.Axes3D(fig)
#ax3d.scatter(k, b, f_kb, c='r')
#ax3d.set_xlabel('k')
#ax3d.set_ylabel('b')
#ax3d.set_zlabel('ErrorArray')
#plt.show()

# 将（ki,bi,f(ki,bi)）连起来，形成曲面
ax = plt.subplot(projection='3d')
ax.plot_surface(k, b, f_kb, rstride=1, cstride=1)
ax.set_xlabel('k')
ax.set_ylabel('b')
ax.set_zlabel('ErrorArray')
plt.show()

# 上面讲了一种简单到夸张的情况，不过我认为很好的理解了mgrid。
# 事实上当Err=∑{i=1~n}（[yi-（k*xi+b）]^2）时也是同样的道理（这是最小二乘法拟合y=kx+b时的误差矩阵）。
# mgrid中第三个参数越大，说明某一区间被分割得越细，相应的曲面越精准。
# 在上面的例子中第三个参数为3j，如果说我们其它不变，单纯将参数改成10j，则曲面图如下：