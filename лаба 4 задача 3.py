import math
import matplotlib.pyplot as plt
import numpy as np
import матрицы
import sys
sys.setrecursionlimit(2000)

## Задача
T_first = 1 #
e_first = [0.001, 0.001, 0.001]
u0_first = [1, 1, 1]

## Явный метод Эйлера
def f_original(y, a, k):
   #a = 1(0.25)2
   #k = 2(0.25)3
   f = [y[1]*y[2]*(k-a)/a ,y[0]*y[2]*(k+a)/k, y[0]*y[1]*(a-k)/a ]
   return f

## Явный метод Эйлера
def search_step(e, f, t_max):
   t_k = []
   for i in range(len(f)):
      t_k.append( e[i] / ( abs(f[i]) + e[i] / t_max ) )
   answer = t_k[0]
   for i in range(len(t_k)):
      if answer > t_k[i]: answer = t_k[i]
   if answer > t_max: answer = t_max
   return answer


def explicit_euler_method(X, Y, Z, U, t, y, tau, T, a, k):
   X.append(t)
   Y.append(y[0])
   Z.append(y[1])
   U.append(y[2])
   f_ = f_original(y, a, k)
   for i in range(len(y)):
      y[i] = y[i] + tau*f_[i]
   t = t + tau
   #
   #print(t)
   #print(y)
   #
   X.append(t)
   Y.append(y[0])
   Z.append(y[1])
   U.append(y[2])
   if t < T: explicit_euler_method(X, Y, Z, U, t, y, search_step(e_first, f_original(y, a, k), T), T, a, k)

   return
##

## Неявный метод Эйлера
tau_max = 0.01
tau_min = 0.0000001

def f_euler_method(yk1, yk0, tau):##
   a = 1
   k = 2
   f = [ yk1[0] - yk0[0] - tau*(yk1[1]*yk1[2]*(k-a)/a),
         yk1[1] - yk0[1] - tau*(yk1[0]*yk1[2]*(k+a)/k),
         yk1[2] - yk0[2] - tau*(yk1[0]*yk1[1]*(a-k)/a)]
   return f

def J_euler_method(X, tau):##
   a = 1
   k = 2
   j = [[1, -tau*(X[2]*(k-a)/a),-tau*(X[1]*(k-a)/a) ], 
        [-tau*(X[2]*(k+a)/k), 1 ,-tau*(X[0]*(k+a)/k) ],
        [-tau*(X[1]*(a-k)/a), -tau*(X[0]*(a-k)/a), 1 ]]
   return j

def searsh_inaccuracy_euler_method(tau0, tau1, y0, y1, y2):
   e = []
   for i in range (len(y0)):
         e.append( ( -tau1 / ( tau1 + tau0 ) ) * ( y2[i] - y1[i] - ( tau1/tau0 ) * ( y1[i] - y0[i] ) ) )
   return e

def func_euler_method(X, tau):
   x0 = X
   x1 = [0]*len(X)
   for i in range(500):
      a = матрицы.product_matrix_by_vector(матрицы.inverse_matrix(J_euler_method(x0, tau), len(X)), f_euler_method(x0, X, tau), len(X))
      for j in range(len(X)):
         x1[j] = x0[j] - a[j]
      if(матрицы.max_(x0, x1) < 0.001):
         return x1
      x0 = x1.copy()

def three_zone_strategy(tau, e):
   tau_k = []
   for i in range(len(e)):
      if(abs(e[i])>e_first[i]):
         tau_k.append(tau/2)
      if(e_first[i]/4 < abs(e[i]) <= e_first[i]):
         tau_k.append(tau)
      if(abs(e[i])<=e_first[i]/4):
         tau_k.append(2*tau)
   answer = min(tau_k)
   return answer


def implicit_euler_method(X, Y1, Y2, Y3, t, y0, y1, tau0, tau1):
   X.append(t)
   Y1.append(y1[0])
   Y2.append(y1[1])
   Y3.append(y1[2])
   t = t + tau1

   y2 = func_euler_method(y1, tau1)

   e = searsh_inaccuracy_euler_method(tau0, tau1, y0, y1, y2)
   flag = True

   for i in range(len(e)):
      if abs(e[i]) < e_first[i]: flag = False

   if(flag == True): implicit_euler_method(X, Y1, Y2, Y3, t - tau1, y0, y1, tau0, tau1 / 2)
   else: 
      tau2 = three_zone_strategy(tau1, e)

      if tau2 > tau_max: tau2 = tau_max

      if t < T_first : implicit_euler_method(X, Y1, Y2, Y3, t, y1, y2, tau1, tau2)
      else: 
         X.append(t)
         Y1.append(y2[0])
         Y2.append(y2[1])
         Y3.append(y1[2])

## Метод Шихмана
def f_shihman_method(yk1, t, yk, yk_1, tau0, tau1):
   w = 20 # 20(1)48
   a = 2.5 + w/40
   a0 = pow(tau1+tau0,2)/(tau0*(2*tau1+tau0))
   a1 = -pow(tau1,2)/(tau0*(2*tau1+tau0))
   b0 = (tau1*(tau1+tau0))/(2*tau1+tau0)
   f = [ yk1[0] - a1*yk[0] - a0*yk_1[0] - b0*(-yk1[0]*yk1[1] + np.sin(t)/t),
         yk1[1] - a1*yk[1] - a0*yk_1[1] - b0*(-pow(yk1[1], 2) + a * t / (1 + pow(t, 2)))]
   return f

def J_shihman_method(X, tau0, tau1):
   b0 = (tau1*(tau1+tau0))/(2*tau1+tau0)
   j = [[1-b0*(-X[1]), 1-b0*(-X[0])], 
        [0, 1-b0*(-2*X[1])]]
   return j

def searsh_inaccuracy_shihman_method(tau_1, tau0, tau1, y_1, y0, y1, y2):
   e = []
   a = pow(tau1, 2)*pow(tau1+tau0,2)/(6*(2*tau1+tau0))
   for i in range (len(y0)):
         e.append( a*6*( y2[i]/(tau1*(tau1+tau0)*(tau1+tau0+tau_1)) - y1[i]/(tau1*tau0*(tau0*tau_1)) + y0[i]/(tau0*tau_1*(tau1*tau0)) - y_1[i]/(tau_1*(tau0+tau_1)*(tau1+tau0+tau_1)) ) )
   return e

def func_shihman_method(X, t, y0, tau0, tau1):
   x0 = X
   x1 = [0]*len(X)
   for i in range(500):
      a = матрицы.product_matrix_by_vector(матрицы.inverse_matrix(J_shihman_method(x0, tau0, tau1), len(X)), f_shihman_method(x0, t, X, y0, tau0, tau1), len(X))
      for j in range(len(X)):
         x1[j] = x0[j] - a[j]
      if(матрицы.max_(x0, x1) < 0.001):
         return x1
      x0 = x1.copy()
   print("ERROR")

def shihman_method(X, Y1, Y2, Y3, index, t, y_1, y0, y1, tau_1, tau0, tau1): # tau1 это тау к-ое, tau0 - тау к-1-ое, tau_1 - тау к-2-ое
   if(index <= 3):
      t = t + tau1

      y2 = func_euler_method(y1, tau1)

      e = searsh_inaccuracy_euler_method(tau0, tau1, y0, y1, y2)
      flag = True

      for i in range(len(e)):
         if abs(e[i]) < e_first[i]: flag = False

      if(flag == True): shihman_method(X, Y1, Y2, Y3, index,  t - tau1, y_1, y0, y1, tau_1, tau0, tau1 / 2)
      else: 
         X.append(t)
         Y1.append(y1[0])
         Y2.append(y1[1])
         Y3.append(y1[2])
         tau2 = three_zone_strategy(tau1, e)

         if tau2 > tau_max: tau2 = tau_max

         if t < T_first: shihman_method(X, Y1, Y2, Y3, index, t, y0, y1, y2, tau0, tau1, tau2)

   elif index > 3:
      t = t + tau1
      y2 = func_shihman_method(y1, t, y0, tau0, tau1)

      e = searsh_inaccuracy_shihman_method(tau_1, tau0, tau1, y_1, y0, y1, y2)
      flag = True
      for i in range(len(e)):
         if abs(e[i]) < e_first[i]: flag = False
      
      if(flag == True): shihman_method(X, Y1, Y2, Y3, index + 1,  t - tau1, y_1, y0, y1, tau_1, tau0, tau1 / 2)
      else: 
         X.append(t)
         Y1.append(y1[0])
         Y2.append(y1[1])
         Y3.append(y1[2])
         tau2 = three_zone_strategy(tau1, e)

         if tau2 > tau_max: tau2 = tau_max

         if t < T_first: shihman_method(X, Y1, Y2, index + 1, t, y0, y1, y2, tau0, tau1, tau2)

##

def main(b):
   if b == 1:
       
      X1 = []
      Y11 = []
      Y21 = []
      Y31 = []

      t1 = 0.000001
      y1 = u0_first.copy()

      explicit_euler_method(X1, Y11, Y21, Y31, t1, y1, search_step(e_first, f_original(y1, 2.5, 2), T_first), T_first, 1, 2)

      X2 = []
      Y12 = []
      Y22 = []
      Y32 = []

      t2 = 0.000001
      y2 = u0_first.copy()

      implicit_euler_method(X2, Y12, Y22, Y32, t2, y2, y2, tau_min, tau_min)

      X3 = []
      Y13 = []
      Y23 = []
      Y33 = []

      t3 = 0.000001
      y3 = u0_first.copy()
      index = 1

      shihman_method(X3, Y13, Y23, Y33, index, t3, y3, y3, y3, tau_min, tau_min, tau_min)

      fig, ax=plt.subplots()
      ax.axhline(y=0, color='k')
      plt.xlabel('Ось Ох')
      plt.ylabel('Ось Оу')
      ax.grid()

      plt.plot(X1, Y11, 'r', label='Явный метод Эйлера U1')
      plt.plot(X1, Y21, 'r-.', label='Явный метод Эйлера U2')
      plt.plot(X1, Y31, 'r--', label='Явный метод Эйлера U3')

      plt.plot(X2, Y12, 'b', label='НЕ Явный метод Эйлера U1')
      plt.plot(X2, Y22, 'b-.', label='НЕ Явный метод Эйлера U2')
      plt.plot(X2, Y32, 'b--', label='НЕ Явный метод Эйлера U3')

      plt.plot(X3, Y13, 'k', label='Метод Шихмана U1')
      plt.plot(X3, Y23, 'k-.', label='Метод Шихмана U2')
      plt.plot(X3, Y33, 'k--', label='Метод Шихмана U3')

      plt.rc('font',family='Times New Roman')
      #plt.legend(loc='best', fontsize = 14,)
      ax.legend(bbox_to_anchor=(0.5, 0.6), fontsize = 14)
      plt.show()
   if b == 2:
      a1 = 1
      k1 = 2
      a2 = 1.5
      k2 = 2.5

      X1 = []
      Y11 = []
      Y21 = []
      Y31 = []

      t1 = 0.000001
      y1 = u0_first.copy()

      explicit_euler_method(X1, Y11, Y21, Y31, t1, y1, search_step(e_first, f_original(y1, a1, k1), T_first), T_first, a1, k1)

      X2 = []
      Y12 = []
      Y22 = []
      Y32 = []

      t2 = 0.000001
      y2 = u0_first.copy()

      explicit_euler_method(X2, Y12, Y22, Y32, t2, y2, search_step(e_first, f_original(y2, a2, k2), T_first), T_first, a2, k2)

      fig, ax=plt.subplots()
      ax.axhline(y=0, color='k')
      plt.xlabel('Ось Ох')
      plt.ylabel('Ось Оу')
      ax.grid()

      #Ввод на экран графика 

      plt.plot(X1, Y11, 'r', label=''.join(['U1 с параметром a=',str(a1),", k=", str(k1)]))
      plt.plot(X1, Y21, 'r--', label=''.join(['U2 с параметром a=',str(a1),", k=", str(k1)]))
      plt.plot(X1, Y31, 'r-.', label=''.join(['U3 с параметром a=',str(a1),", k=", str(k1)]))

      plt.plot(X2, Y12, 'b--', label=''.join(['U1 с параметром a=',str(a2),", k=", str(k2)]))
      plt.plot(X2, Y22, 'b', label=''.join(['U2 с параметром a=',str(a2),", k=", str(k2)]))
      plt.plot(X2, Y32, 'b-.', label=''.join(['U3 с параметром a=',str(a2),", k=", str(k2)]))



      plt.rc('font',family='Times New Roman')
      #plt.legend(loc='best', fontsize = 14)
      ax.legend(bbox_to_anchor=(0.5, 0.4), fontsize = 14)
      plt.show()

main(2)