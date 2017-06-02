import numpy as np
import time

def numpysum(n):
	a = np.arange(n)**2
	b = np.arange(n)**3
	c = a + b
	return c

def pythonsum(n):
	a = range(n)
	b = range(n)
	c = []
	for i in range (n):
		a[i] = i**2
		b[i] = i**3
		c.append(a[i] + b[i])
	return c

tsum_1 = 0
for n in range(5000):
	start = time.time()
	numpysum(n)
	delta = time.time() - start
	if n % 100 == 0 and n != 0:
		print "%d delta: %f" % (n, delta)
	tsum_1 += delta

tsum_2 = 0
for n in range(5000):
	start = time.time()
	pythonsum(n)
	delta = time.time() - start
	if n % 100 == 0 and n != 0:
		print "%d delta: %f" % (n, delta)
	tsum_2 += delta

print "numpysum ", tsum_1
print "pythonsum ", tsum_2
