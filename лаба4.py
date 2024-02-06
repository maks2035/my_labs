import math
import matplotlib.pyplot as plt
import numpy as np

matrix = [[0.14, 0.24, -0.84],[1.07, -0.83, 0.56],[0.64, 0.43, -0.38]]
matrix_answers = [1.11, 0.48, -0.83]

matrix1 = [[1.4, 0.24, -0.84],[1.07, -8.3, 0.56],[0.64, 0.43, -3.8]]
matrix_answers1 = [1.11, 0.48, -0.83]

X = [matrix_answers1[0]/matrix1[0][0], matrix_answers1[1]/matrix1[1][1], matrix_answers1[2]/matrix1[2][2]]
X1 = [1, 1, 1]
X2 = [1, 1, 1]
flag = True

def calculation_new_X_p_i(X_pref, A, B):
   X = []
   X.append((-A[0][1]*X_pref[1]-A[0][2]*X_pref[2]+B[0])/A[0][0])
   X.append((-A[1][0]*X_pref[0]-A[1][2]*X_pref[2]+B[1])/A[1][1])
   X.append((-A[2][0]*X_pref[0]-A[2][1]*X_pref[1]+B[2])/A[2][2])
   return X

def calculation_new_X_z(X_pref, A, B):
   X = []
   X.append((-A[0][1]*X_pref[1]-A[0][2]*X_pref[2]+B[0])/A[0][0])
   X.append((-A[1][0]*X[0]-A[1][2]*X_pref[2]+B[1])/A[1][1])
   X.append((-A[2][0]*X[0]-A[2][1]*X[1]+B[2])/A[2][2])
   return X

def determine_applicability(A):
   a = sum(abs(a/A[0][0]) for a in A[0])
   if a < sum(abs(a/A[1][1]) for a in A[1]): a = sum(abs(a/A[1][1]) for a in A[1])
   if a < sum(abs(a/A[2][2]) for a in A[2]): a = sum(abs(a/A[2][2]) for a in A[2])
   a -= 1
   if a < 1: return 1 
   return 0


if determine_applicability(matrix1) == 1:
   print("метод простых итераций:")
   for i in range(10):
      X_pref1 = X1
      X1 = calculation_new_X_p_i(X1, matrix1, matrix_answers1)
      print(X1)
      E1 = [0, 0, 0]
      for j in range(3): E1[j] = X_pref1[j] - X1[j]
      e1 = E1[0]
      for k in range(3): 
         if abs(e1) < abs(E1[k]): e1 = E1[k] 
      if(abs(e1) < 0.001): break

   print("Метод Гаусса — Зейделя:")
   for i in range(10):
      X_pref2 = X2
      X2 = calculation_new_X_z(X2, matrix1, matrix_answers1)
      print(X2)
      E2 = [0, 0, 0]
      for j in range(3): E2[j] = X_pref2[j] - X2[j]
      e2 = E2[0]
      for k in range(3): 
         if abs(e2) < abs(E2[k]): e2 = E2[k] 
      if(abs(e2) < 0.001): break
