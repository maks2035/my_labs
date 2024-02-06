import math
import copy

#вывод матрицы
def print_matrixes(matrix, matrix_answers, rows):
   print('\n', end='')
   for i in range (rows):
      print(matrix[i], matrix_answers[i])
   print('\n', end='')

#заполнение матрицы
def filling_matrix(matrix, rows, columns):
   for i in  range(rows):
         for j in  range(columns):
            print("введите элемент %d строки %d столбца" %(i + 1, j + 1))
            matrix[i][j] = float(input())

#заполнение стобца "ответов"
def filling_matrix_answers(matrix_answers, rows): 
   for i in  range(rows):
      print("введите %d ответ" %(i + 1))
      matrix_answers.append(float(input()))

#перемена местави 2-х строк
def swap_rows (matrix, matrix_answers, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
    matrix_answers[row1], matrix_answers[row2] = matrix_answers[row2], matrix_answers[row1]

#деление строки на число
def divide_row (matrix, matrix_answers, row, divider):
    matrix[row] = [a / divider for a in matrix[row]]
    matrix_answers[row] /= divider

#сложение строки с другой строкой, умноженной на число
def combine_rows (matrix, matrix_answers, row, source_row, weight):
    matrix[row] = [(a + k * weight) for a,k in zip(matrix[row], matrix[source_row])]
    matrix_answers[row] += matrix_answers[source_row] * weight

#нахождение строки с максимальным по модулю элементом в толбце 
def find_max_elem_in_column(matrix, rows, current_rows, current_columns):
   index_max_elem = current_rows
   for i in range(current_rows, rows):
      if index_max_elem is abs(matrix[i][current_columns]) > abs (matrix[index_max_elem][current_columns]):
         index_max_elem = i
   return index_max_elem


"""
print("введите количество столбцов")
columns = int(input())

print("введите количество строк")
rows = int(input())

while rows < columns :
   print("попробуйде еще раз")
   rows = int(input())

"""
rows = 3
columns = 3

matrix = [[0.0] * columns for i in range(rows)]
matrix_answers = []


matrix = [[0.14, 0.24, -0.84],[1.07, -0.83, 0.56],[0.64, 0.43, -0.38]]
matrix_answers = [1.11, 0.48, -0.83]
"""

filling_matrix(matrix, rows, columns)
filling_matrix_answers(matrix_answers, rows)
"""
print_matrixes(matrix, matrix_answers, rows)


A = copy.deepcopy(matrix)
B = copy.deepcopy(matrix_answers)
for i in range(columns):
   max_matrix_index = find_max_elem_in_column(A, rows, i, i)
   swap_rows(A, B, i, max_matrix_index)
   divide_row(A, B, i, A[i][i])
   print_matrixes(A, B, rows)
   for j in range(i+1, rows):
      combine_rows(A, B, j, i, -A[j][i])
      print_matrixes(A, B, rows)
   print_matrixes(A, B, rows)
   if ((all(a == 0 for a in A[i])) and (B[i] != 0)):
      print("Error")
      exit(0)

X = [0 for b in B]
for i in range(len(B)-1, -1, -1):
   X[i] = B[i] - sum(x*a for x,a in zip(X[(i+1):], A[i][(i+1):]))
print("вектор неизвестных")
print(X)

Y = [0 for b in matrix_answers]
for i in range(rows):
   for j in range(columns):
      Y[i] += matrix[i][j]*X[j]
   Y[i] -= matrix_answers[i]
print("вектор невязки")
print(Y)
