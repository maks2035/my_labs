import math
import matplotlib.pyplot as plt
import numpy as np

def Teylor(x, k):
    result = 0.0
    n = 0
    while n <= k:
        result = result + ( ( math.pow(-1, n) * math.pow( x , ( 2 * n + 1 ) ) ) / math.factorial( 2 * n + 1 ) )
        result = float('{:.5f}'.format(result))
        n += 1
    return result

summ14 = []

for x in np.arange(-4*np.pi,4*np.pi+0.1, 0.1):
    summ14.append(Teylor(x, 14))
summ16 = []

for x in np.arange(-4*np.pi,4*np.pi+0.1, 0.1):
    summ16.append(Teylor(x, 16))

summ19 = []

for x in np.arange(-4*np.pi,4*np.pi+0.1, 0.1):
    summ19.append(Teylor(x, 19))

# 100 linearly spaced numbers
x = np.linspace(-4*np.pi,4*np.pi, 253)

# the function, which is y = sin(x) here
y = np.sin(x)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the function
plt.plot(x,y, 'r-', label='y=sin(x)')
plt.plot(x,summ14, 'k--', label='Разложение в ряд Тейлора с 14 членами')
plt.plot(x,summ16, 'b--', label='Разложение в ряд Тейлора с 16 членами')
plt.plot(x,summ19, 'g--', label='Разложение в ряд Тейлора с 19 членами')

plt.legend(loc='upper left')

# show the plot
plt.show()