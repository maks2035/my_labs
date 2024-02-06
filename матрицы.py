import math
import numpy as np
import matplotlib.pyplot as plt


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
