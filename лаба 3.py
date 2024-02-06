import math
import matplotlib.pyplot as plt
import numpy as np

ANSWER = [0.7657010466582012, 0.5306073406165416, 0.363535083905772]

x0 = [1, -1, 1]

# исходная система
def f(X):
   f = [X[0]*X[0] + X[1]*X[1] + X[2]*X[2]- 1, 2*X[0]*X[0] + X[1]*X[1] - 4 * X[2], 3*X[0]*X[0] - 4 * X[1] + X[2]]
   return f

# якобиан
def J(X):
   j = [[2*X[0], 2*X[1], 2*X[2],], [4*X[0], 2*X[1], -4], [6*X[0], -4, 1]]
   return j

#нахождение минора
def minor(A, n, i, j):
   di = 0
   dj = 0

   matrix = [[0]*n for i in range(n)]

   for ki in range(n-1):
      if(ki == i): di = 1
      dj = 0
      for kj in range(n-1):
         if(kj == j): dj = 1
         matrix[ki][kj] = A[ki+di][kj+dj]
   return matrix

#нахождение определителя
def det(A, n):
   d = 0
   k = 1
   if (n == 1):
      return A[0][0]
   
   if(n == 2):
        return A[0][0] * A[1][1] - (A[1][0] * A[0][1])
    
   if(n > 2):
      for i in range(n):
         m = minor(A, n, i, 0)
         d = d + k * A[i][0] * det(m, n-1)
         k = -k
   
   return d

# обратная матрица
def inverse_matrix(A, n):
   det_A = det(A, n)
   if(det_A == 0):
      print("ошибка")
      exit(0)
   matrix = [[0]*n for i in range(n)]

   for i in range(n):
       for j in range(n):
           matrix[j][i] = pow(-1, i+j) * det(minor(A, n, i, j), n-1) /det_A
   
   return matrix

# произведение матрицы на вектор
def product_matrix_by_vector(A, v , n):
   vect = [0]*n

   for i in range(n):
       for j in range(n):
           vect[i] = vect[i] + A[i][j]*v[j]

   return vect
# максимальная разница  между элементами 2-х векторов
def max_(x1, x2):
   max = 0
   for i in range(len(x2)):
      if(abs(x2[i] >= 1)):
         if(abs((x1[i]-x2[i])/x2[i])> max): max = abs((x1[i]-x2[i])/x2[i])
      else:
         if(abs(x1[i]-x2[i])> max): max = abs(x1[i]-x2[i])
   return max

# сам итерационнаый процесс x, y, z используются для графика(не обязательны) 
def func(X, x, y, z):
   x0 = X
   x1 = [0]*len(X)
   for i in range(501):
      if i == 500: 
         print("ERROR")
         return
      print("итерация ", i, ": ", x0)
      x.append(x0[0]) # нужно для графика(можно убрать)
      y.append(x0[1]) # нужно для графика(можно убрать)
      z.append(x0[2]) # нужно для графика(можно убрать)
      a = product_matrix_by_vector(inverse_matrix(J(x0), len(X)), f(x0), len(X))
      for j in range(len(X)):
         x1[j] = x0[j] - a[j]
      #print(x0, " - ", inverse_matrix(J(x0), len(X)), " * ", f(x0), " = ", x1 )
      if(max_(x0, x1) < 0.001):
         print("итерация ", i+1, ": ", x1)
         x.append(x1[0])# нужно для графика(можно убрать)
         y.append(x1[1])# нужно для графика(можно убрать)
         z.append(x1[2])# нужно для графика(можно убрать)
         return
      x0 = x1.copy()

a = int(input())

x = []
y = []
z = []

func(x0, x, y, z)

#3д поверхности (сосвсем не обязательно)
u1 = np.linspace(-np.pi, np.pi, 100)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)


#outer - Вычислите внешнее произведение двух векторов.
#ones - Возвращает новый массив заданной формы и типа, заполненный единицами.

x1 = np.outer(np.cos(u), np.sin(v))
y1 = np.outer(np.sin(u), np.sin(v))
z1 = np.outer(np.ones(np.size(u)), np.cos(v))

print(x1)
def function_z(x,y):
   return (x**2)/2 + (y**2)/4

#meshgrid - Возвращает координатные матрицы из координатных векторов.
x2, y2 = np.meshgrid(u1, u1)
z2 = (x2**2)/4 + (y2**2)/4


x3, y3 = np.meshgrid(u1, u1)
z3 = -3*(x3**2) + 4*y3
#(сосвсем не обязательно кончилось)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if(a == 1):
   for i in range(len(x)):
      ax.scatter(x[i],y[i],z[i], marker = ".")

   plt.xlim(-2, 2)
   plt.ylim(-2, 2)
   ax.set_zlim(-2, 2)

   # Plot the values
   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.set_zlabel('Z-axis')

   plt.rc('font',family='Times New Roman')
   #plt.legend(loc='upper right', fontsize = 14)
   plt.show()

#3д поверхности (сосвсем не обязательно)
if(a == 2):
   ax.plot_surface(x1, y1, z1, alpha = 0.3)
   ax.plot_surface(x2, y2, z2, alpha = 0.3)
   ax.plot_surface(x3, y3, z3, alpha = 0.3)

   plt.xlim(-2, 2)
   plt.ylim(-2, 2)
   ax.set_zlim(-2, 2)

   # Plot the values
   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.set_zlabel('Z-axis')

   plt.rc('font',family='Times New Roman')
   #plt.legend(loc='upper right', fontsize = 14)
   plt.show()
#(сосвсем не обязательно кончилось)

if(a == 3):
   for i in range(len(x)):
      ax.scatter(x[i],y[i],z[i], marker = ".")
   
   ax.plot_surface(x1, y1, z1, alpha = 0.3)
   ax.plot_surface(x2, y2, z2, alpha = 0.3)
   ax.plot_surface(x3, y3, z3, alpha = 0.3)

   plt.xlim(-2, 2)
   plt.ylim(-2, 2)
   ax.set_zlim(-2, 2)

   # Plot the values
   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.set_zlabel('Z-axis')

   plt.rc('font',family='Times New Roman')
   #plt.legend(loc='upper right', fontsize = 14)
   plt.show()