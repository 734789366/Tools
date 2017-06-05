# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 在Numpy中，矩阵是ndarray的子类。创建矩阵，可由mat, matrix, bmat来实现
a = np.mat('1 2 3; 4 5 6; 7 8 9')
print "a:", a
print "transpose A", a.T
# print "Invert A", a.I
b = np.mat(np.arange(9).reshape(3, 3))
print "create from NumPy:", b
print "type(b)", type(b)

# 利用一些已有的较小的矩阵来创建一个新的大矩阵。这可以用bmat函数来实现。这里的 b 表示“分块”
# bmat即分块矩阵(block matrix). 使用字符串创建复合矩阵,该字符串的格式与mat函数中一致
A = np.eye(2)
B = 2 * A
print A, B
print "复合矩阵", np.bmat('A B;B A')

# 我们可以使用NumPy中的frompyfunc函数,通过一个Python函数来创建通用函数.
def ultimate_answer(a):
    result = np.zeros_like(a)
    result.fill(42)
    return result
ufunc = np.frompyfunc(ultimate_answer, 1, 1)
print "the answer:", ufunc(np.arange(4).reshape(2, 2))

# 其实通用函数并非真正的函数,而是能够表示函数的对象。通用函数有四个方法,
# 不过这些方法只对输入两个参数、输出一个参数的ufunc对象有效, 例如add函数,
# 其他不符合条件的ufunc对象调用这些方法时将抛出ValueError异常。因此只能
# 在二元通用函数上调用这些方法.这4个方法:
# reduce
# accumulate
# reduceat
# outer

# 沿着指定的轴,在连续的数组元素之间递归调用通用函数,即可得到输入数组的规约(reduce)计算结果
a = np.arange(9)
print "reduce:", np.add.reduce(a)

# accumulate方法同样可以递归作用于输入数组。但是与reduce方法不同的是,它将存储运算的中间结果并返回
print "accumulate:", np.add.accumulate(a)

# reduceat方法解释起来有点复杂
print "reduceat", np.add.reduceat(a, [0, 5, 2, 7])

# outer 方法返回一个数组,它的秩(rank)等于两个输入数组的秩的和。它会作用于两个输入数组之间存在的所有元素对
print "Outer:", np.add.outer(a, np.arange(3))
print "Outer:", np.add.outer(np.arange(3), a)

# 在NumPy中,基本算术运算符+、-和 * 隐式关联着通用函数add,subtract和multiply,
# 也就是说,当你对NumPy数组使用这些算术运算符时,对应的通用函数将自动被调用.
# 在数组的除法运算中涉及三个通用函数divide 、true_divide和floor_division,
# 以及两个对应的运算符 / 和 //
# divide 函数在整数和浮点数除法中均只保留整数部分
a = np.array([2, 6, 5])
b = np.array([1, 2, 3])
print "np.divide(a, b):", np.divide(a, b)
print "np.divide(b, a):", np.divide(b, a)

# true_divide函数与数学中的除法定义更为接近,即返回除法的浮点数结果而不作截断
print "np.true_divide(a, b):", np.true_divide(a, b)
print "np.true_divide(b, a):", np.true_divide(b, a)
# floor_divide函数总是返回整数结果,相当于先调用divide函数再调用floor函数.
# floor函数将对浮点数进行向下取整并返回整数.
print "Floor Divide:", np.floor_divide(a, b), np.floor_divide(b, a)
c = 3.14*4
print "Floor Divide 2:", np.floor_divide(c, b), np.floor_divide(b, c)

# 默认情况下,使用/运算符相当于调用divide函数
# from__future__import division
# 但如果在Python程序的开头有上面那句代码,则改为调用 true_divide 函数
print "/ operator: ", a/b, b/a
# 运算符//对应于floor_divide函数
print "// operator: ", a//b, b//a
print "// operator 2: ", c//b, b//c

# 计算模数或者余数,可以使用NumPy中的mod、remainder和fmod函数。当然,也可以使用%运算符
# remainder函数逐个返回两个数组中元素相除后的余数。如果第二个数字为0,则直接返回0
# mod函数与remainder函数的功能完全一致.
# %操作符仅仅是remainder函数的简写
a = np.arange(-4, 4)
print "Remainder: ", np.remainder(a, 2)
print "mod: ", np.mod(a, 2)
print "%: ", a%2
# fmod函数处理负数的方式与remainder、mod和%不同。所得余数的正负由被除数决定,与除数的正负无关
print "fmod: ", np.fmod(a, 2)

# 斐波那契(Fibonacci)数列是基于递推关系生成的, 斐波那契数列的递推关系可以用矩阵来表示
# 斐波那契数列的计算等价于矩阵的连乘.
F = np.matrix([[1, 1], [1, 0]])
print "F: ", F
# 计算斐波那契数列中的第8个数,即矩阵的幂为7减去1。计算出的斐波那契数位于矩阵的对角线上
print "8th Fibonacci:", (F**7)[0, 0]
# 利用黄金分割公式或通常所说的比奈公式(Binet’ s Formula),加上取整函数,就可以直接计算斐波那契数
# ront()函数对浮点数取整,但结果仍为浮点数类型.
n = np.arange(1, 9)
sqrt5 = np.sqrt(5)
phi = (1 + sqrt5)/2
fibonacci = np.rint((phi**n - (-1/phi)**n)/sqrt5)
print "Fibonacci: ", fibonacci

def calc_fibonacci(a):
    n = np.arange(1, a+1)
    sqrt5 = np.sqrt(5)
    phi = (1 + sqrt5)/2
    fibonacci = np.rint((phi**a - (-1/phi)**n)/sqrt5)
    return fibonacci[-1]

fibonacci = np.frompyfunc(calc_fibonacci, 1, 1)
for i in range(1, 9):
    print "fibonacci(%d): %d" % (i, fibonacci(i))

# 在NumPy中,所有的标准三角函数如sin、cos、tan等均有对应的通用函数.
# 利萨茹曲线(Lissajous curve)是一种很有趣的使用三角函数的方式.
# 利萨茹曲线由以下参数方程定义:
# x = A sin(at + np.pi/2)
# y = B sin(bt)
a = 9
b = 8
t = np.linspace(-np.pi, np.pi, 201)
x = np.sin(a*t + np.pi/2)
y = np.cos(b*t)
plt.plot(x, y, label=u'利萨茹曲线')
plt.show()

# 方波也是一种可以在示波器上显示的波形。方波可以近似表示为多个正弦波的叠加。事实上,
# 任意一个方波信号都可以用无穷傅里叶级数来表示
t = np.linspace(-np.pi, np.pi, 201)
k = np.arange(1, 99)
k = 2*k - 1
f = np.zeros_like(t)
for i in range(len(t)):
    f[i] = np.sum(np.sin(k*t[i])/k)
f = 4 * f / np.pi
plt.plot(t, f, label=u'方波')
plt.show()

k = np.arange(1, 99)
k = np.pi * k
for i in range(len(t)):
	f[i] = np.sum(np.sin(2*k*t[i])/k)
f = -2*f
plt.plot(t, f, 'r-', label=u'三角波')
plt.plot(t, np.abs(f), 'g-', label=u'锯齿波')
plt.show()

# 位操作函数可以在整数或整数数组的位上进行操作,它们都是通用函数.
# ^ 、 & 、 | 、 << 、 >> 等位操作符在NumPy中也有对应的部分,
# < 、 > 、 == 等比较运算符也是如此
# 检查两个整数的符号是否一致
# 因此当两个操作数的符号不一致时, XOR 操作的结果为负数。在NumPy中,
# ^ 操作符对应于 bitwise_xor 函数,< 操作符对应于 less 函数
x = np.arange(-9, 9)
y = -x
print "sign different?", x^y < 0
print "sign different?", np.less(np.bitwise_xor(x, y), 0)

# 检查一个数是否为2的幂数
# 在二进制数中,2的幂数表示为一个1后面跟一串0的形式,例如 10 、 100 、 1000 等.
# 而比2的幂数小1的数表示为一串二进制的1,例如 11 、 111 、 1111 (即十进制里的3、7、15)等.
# 如果我们在2的幂数以及比它小1的数之间执行位与操作 AND ,那么应该得到0.
print "power of 2?", (x&(x-1) == 0)
print "power of 2?", np.equal(np.bitwise_and(x, x-1), 0)

# 计算一个数被2的幂数整除后的余数
# 计算余数的技巧实际上只在模为2的幂数(如4、8、16等)时有效。二进制的位左移一位,
# 则数值翻倍。在前一个小技巧中我们看到,将2的幂数减去1可以得到一串1组成的二进制数,如
# 11 、 111 、 1111 等。这为我们提供了掩码(mask),与这样的掩码做位与操作 AND 即可得到以2
# 的幂数作为模的余数。在NumPy中, << 操作符对应于 left_shift 函数
print "4的余数", x & ((1<<2)-1)
print "4的余数", np.bitwise_and(x, np.left_shift(1, 2)-1)