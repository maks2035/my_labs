import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):#начальная функция 
   f = math.sin(x) - x + 3
   return f

def f1(x):#первая производная
   f = math.cos(x) - 1
   return f

def method_chego_to(X, x1, x2):
   print("[",x1,";",x2,"]")
   X.append(x1)
   X.append(x2)
   if(abs(x1-x2) < 0.01): return 
   x3 = (x1+x2)/2
   if(f(x1)*f(x3) < 0):
      method_chego_to(X, x1, x3)
   elif(f(x2)*f(x3) < 0):
      method_chego_to(X, x3, x2)

def method_newton(X, x0, a):
   if(f1(x0) == 0):
      print("корректное значение производной")
      exit(0)
   x1 = x0 - f(x0)/f1(x0)
   if (a == False): 
      print(x0)
      X.append(x0)
   print(x1)
   X.append(x1)
   if(abs(x0 - x1) < 0.01): return
   method_newton(X, x1, a+1)

def method_chord(X, x0, x1, a):
   if (a == False): 
      print(x0)
      X.append(x0)
   print(x1)
   X.append(x1)
   if(abs(x0 - x1) < 0.01): return
   x2 = x0 - (f(x0)/(f(x1)-f(x0)))*(x1-x0)
   method_chord(X, x1, x2, a+1)

def method_secant(X, x0, x1, a):
   if (a == False): 
      print(x0)
      X.append(x0)
   print(x1)
   X.append(x1)
   if(abs(x0 - x1) < 0.01): return
   x2 = x1 - (f(x1)/(f(x0)-f(x1)))*(x0-x1)
   method_secant(X, x1, x2, a+1)

def simple_iterations(X, x0, a):
   k = -2
   x1 = x0 - f(x0)/k
   if (a == False): 
      print(x0)
      X.append(x0)
   print(x1)
   X.append(x1)
   if(abs(x0 - x1) < 0.01): return
   simple_iterations(X, x1, a+1)

fig, ax=plt.subplots()
ax.axhline(y=0, color='k')


a = int(input())#выбор какой метод использовать

if(a == 0):
   x = []
   y = []
   for i in np.arange(-4*np.pi,4*np.pi, 0.1):
      x.append(i)
      y.append(f(i))
   ax.grid()
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='lower left', fontsize = 14)
   plt.show()

if(a == 1):
   x = []
   y = []
   for i in np.arange(0,6, 0.1):
      x.append(i)
      y.append(f(i))
   X = []
   method_chego_to(X, 0, 6)
   Y = [0]*len(X)
   ax.plot(X,Y, 'bo', label='полученные приближенные значения')
   ax.grid()
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='lower left', fontsize = 14)
   plt.show()

if(a == 2):
   x = []
   y = []
   for i in np.arange(0,6, 0.1):
      x.append(i)
      y.append(f(i))
   X = []
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   ax.grid()
   method_newton(X, 1.5, 0)
   for i in range(len(X)-1):
      X_temp = []
      X_temp.append(X[i])
      X_temp.append(X[i+1])
      Y_temp = []
      Y_temp.append(f(X[i]))
      Y_temp.append(0)
      if(i == 0):
         plt.plot(X_temp,Y_temp, 'b', label='Найденные косательные')
      else:
         plt.plot(X_temp,Y_temp, 'b')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='lower left', fontsize = 14)
   plt.show()

if(a == 3):
   x = []
   y = []
   for i in np.arange(0.5,4.1, 0.1):
      x.append(i)
      y.append(f(i))
   X = []
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   ax.grid()
   method_chord(X, 1, 4, 0)
   for i in range(1, len(X)):
      X_temp = []
      X_temp.append(X[0])
      X_temp.append(X[i])
      Y_temp = []
      Y_temp.append(f(X[0]))
      Y_temp.append(f(X[i]))
      if(i == 1):
         plt.plot(X_temp,Y_temp, 'b', label='Полученные хорды')
      else:
         plt.plot(X_temp,Y_temp, 'b')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='lower left', fontsize = 14)
   plt.show()

if(a == 4):
   x = []
   y = []
   for i in np.arange(0.5,4.1, 0.1):
      x.append(i)
      y.append(f(i))
   X = []
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   ax.grid()
   method_secant(X, 1, 4, 0)
   for i in range(0, len(X)-1):
      X_temp = []
      X_temp.append(X[i])
      X_temp.append(X[i+1])
      Y_temp = []
      Y_temp.append(f(X[i]))
      Y_temp.append(0)
      if(i == 1):
         plt.plot(X_temp,Y_temp, 'b', label='Полученные секущие')
      else:
         plt.plot(X_temp,Y_temp, 'b')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='lower left', fontsize = 14)
   plt.show()

if(a == 5):
   x = []
   y = []
   for i in np.arange(0,6, 0.1):
      x.append(i)
      y.append(f(i))
   X = []
   plt.plot(x,y, 'r', label='изначальный график')
   plt.xlabel('Ось Ох')
   plt.ylabel('Ось Оу')
   ax.grid()
   simple_iterations(X, 0, 0)
   Y = [0]*len(X)
   ax.plot(X,Y, 'bo', label='полученные приближенные значения')
   plt.rc('font',family='Times New Roman')
   plt.legend(loc='upper right', fontsize = 14)
   plt.show()