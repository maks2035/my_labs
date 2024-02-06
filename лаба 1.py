import math
import copy

#заполнение матрицы
def filling_matrix(n):
   A = [[0]*n for i in range(n)]

   for i in  range(n):
         for j in  range(n):
            print("введите элемент %d строки %d столбца" %(i + 1, j + 1))
            A[i][j] = float(input())
   
   return A

#заполнение матрицы
def filling_vect(n):
   a = [0]*n
   for i in  range(n):
         
         print("введите элемент %d " %(i + 1))
         a[i] = float(input())

   return a

#нахождение минора
def minor(A, n, i, j):
   di = 0
   dj = 0
   #
   matrix = [[0]*(n-1) for i in range(n-1)]
   #
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
   #
   if(n == 1):
      return A[0][0]
   #
   if(n == 2):
      return A[0][0] * A[1][1] - (A[1][0] * A[0][1])
    
   if(n > 2):
      for i in range(n):
         m = minor(A, n, i, 0)
         d = d + k * A[i][0] * det(m, n-1)
         k = -k
   
   return d

#скаляроное произведение
def scalar_product(a, b, n):
   c = 0

   for i in range(n):
      c = c + a[i]*b[i]

   return c

def product_matrix_by_vector(A, v , n):
   vect = [0]*n

   for i in range(n):
       for j in range(n):
           vect[i] = vect[i] + A[i][j]*v[j]

   return vect

def inverse_matrix(A, n):
   det_A = det(A, n)
   matrix = [[0]*n for i in range(n)]

   for i in range(n):
       for j in range(n):
           matrix[j][i] = pow(-1, i+j) * det(minor(A, n, i, j), n-1) / det_A
   
   return matrix

def search_own_number(A, u, n):
   v0 = u
   v1 = product_matrix_by_vector(A, v0, n)
   int1 = scalar_product(v0, v1, n) / scalar_product(v0, v0, n)
   int2 = 0
   i = 1
   while(True):
      v0 = v1
      v1 = product_matrix_by_vector(A, v0, n)
      int2 = scalar_product(v0, v1, n) / scalar_product(v0, v0, n)
      if(abs(int1 - int2) < 0.01): 
         print("количестнов итераций ", i)
         return int2
      int1 = int2
      i += 1


n = 0

while (n < 2):
   print("введите размерность матрицы")
   n = int(input())

A = filling_matrix(n)

if(det(A, n) == 0): 
   print("вырожденная матрица")
   exit()

u = filling_vect(n)
B = inverse_matrix(A, n)

own_int_max = search_own_number(A, u, n)
print(abs(own_int_max))


own_int_min = 1 / search_own_number(B, u, n)
print(abs(own_int_min))