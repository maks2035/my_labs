import math
import matplotlib.pyplot as plt
import numpy as np

X = [5.1, 11.9, 17.5, 23.1, 29.4]
Y =[0.93, 0.79, 0.71, 0.75, 0.91]

A1 = np.array([[len(X), sum(x for x in X), sum(math.pow(x, 2) for x in X) ],
            [sum(x for x in X), sum(math.pow(x, 2) for x in X), sum(math.pow(x, 3) for x in X)],
            [sum(math.pow(x, 2) for x in X), sum(math.pow(x, 3) for x in X), sum(math.pow(x, 4) for x in X)],
            ])
B1 = np.array([sum(y for y in Y), sum(x*y for x , y in zip (X, Y)),sum(math.pow(x, 2)*y for x , y in zip (X, Y)) ])
C1 = np.linalg.solve(A1, B1)
def square (C, x):
   y = C[0] + C[1]*x + C[2] * math.pow(x, 2)
   return y
def line (C, x):
   y = C[0] + C[1]*x
   return y

A2 = np.array([[len(X), sum(x for x in X)], [sum(x for x in X), sum(math.pow(x, 2) for x in X)]])
B2 = np.array([sum(y for y in Y), sum(x*y for x , y in zip (X, Y))])
C2 = np.linalg.solve(A2, B2)

y1 = []
for i in np.arange(0,31, 0.1):
    y1.append(square(C1, i))
y2 = []
for i in np.arange(0,31, 0.1):
    y2.append(line(C2, i))

x = []
for i in np.arange(0,31, 0.1):
    x.append(i)

def standard_deviation_square(C1, X , Y):
    A = []
    for x, y in zip(X , Y): A.append(y-square(C1, x))
    result = pow((sum(pow(a - (sum(a for a in A)/len(A)) , 2) for a in A)/len(A)), 0.5)
    return result

def standard_deviation_line(C2, X, Y):
    A = []
    for x, y in zip(X , Y): A.append(y - line(C2, x))
    result = pow((sum(pow(a - (sum(a for a in A)/len(A)) , 2) for a in A)/len(A)), 0.5)
    return result

print("Среднеквадротические отклонение линейной апроксимации: ", standard_deviation_line(C2, X, Y))
print("Среднеквадротические отклонение квадротичной апроксимации апроксимации: ", standard_deviation_square(C1, X, Y))
plt.axis([0,31,0,1.5])
plt.xlabel('Ось Ох, у.е.')
plt.ylabel('Ось Оу, у.е.')
plt.scatter(X,Y, label='Изначальные значения')
plt.plot(x,y1, "r-", label='Аппроксимация квадратичной функцией')
plt.plot(x,y2, "g-", label='Аппроксимация линенйной функцией')
plt.rc('font',family='Times New Roman')
plt.legend(loc='upper right', fontsize = 14)

plt.show()