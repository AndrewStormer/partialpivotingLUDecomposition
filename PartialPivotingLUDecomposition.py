import numpy as np


def main():
    #matrix_size = int(input("Enter matrix size: "))
    matrix_size = 3
    A = np.zeros((matrix_size, matrix_size))
    temp = [[0, 0.4, 0.6], [0.6, 0.1, 0.2], [0.4, 0.5, 0.2]]
    b = np.zeros(matrix_size)
    for i in range(matrix_size):
        for j in range(matrix_size):
            A[i][j] = temp[i][j] #have to do this becuase python is dumb :)
            
    print("Matrix Equation: ")
    printmatrixequation(A, b, matrix_size)
    interchanges = partialpivotingLUDecomposition(A, matrix_size)
    print("Modified Matrix: \n", A)
    print("Interchanges done: ", interchanges)
    solvePLUsystem(A, b, matrix_size, interchanges)
    print("Vector x solved for in equation Ax = b: \n", b)
    

#Helper functions  
def printmatrixequation(A, b, matrix_size):
    for i in range(matrix_size):
        print("| ", end=" ")
        for j in range(matrix_size):
            print(A[i][j], end=" ")
        print("| * | x[{}] |   =  | ".format(i), b[i], "|\n")

  
def swap(a, b):
    return b, a
    

#LU Decomposition functions    
#Performs Gaussian Elimination with partial pivoting, sorting L and U over A, returning the interchanges
def partialpivotingLUDecomposition(A, matrix_size):
    interchanges = np.zeros(matrix_size)
    flag = 0
    
    for k in range(matrix_size-1):
        amax = 0
        m = 0
        for i in range(k, matrix_size):
            if (abs(A[i][k]) > amax): #find the greatest element in the kth column
                amax = abs(A[i][k])
                m = i #store pivot position
                
        if amax == 0: #matrix A is singular so U must be singlular
            flag = 1
            interchanges[k] = 0
        else:
            interchanges[k] = m #store interchanges (same as storing a 1 in P[k][m] in permutation matrix)
            if m != k:
                for i in range(matrix_size):
                    A[k][i], A[m][i] = swap(A[k][i], A[m][i]) #interchange element i in row k with row m
            
            for i in range(k+1, matrix_size):
                A[i][k] /= A[k][k] #set multipliers in L
                
            for i in range(k+1, matrix_size):
                for j in range(k+1, matrix_size):
                    A[i][j] = A[i][j] - A[i][k]*A[k][j] #row operations
                    
    if A[matrix_size-1][matrix_size-1] == 0: #matrix A is singular so U must be singular
        flag = 1
        interchanges[matrix_size-1] = 0
    else:
        interchanges[matrix_size-1] = matrix_size-1 #store interchanges (same as storing a 1 in P[k][matrix_size-1] in permutation matrix)
        
    if flag == 1:
        print("Matrix A is singular, so U will be singular\n")
    return interchanges


#Solves Ax = b for x given decomposed matrix A, original vector b and a record of the interchanges. b is changed in place to x
def solvePLUsystem(A, b, matrix_size, interchanges):
    for k in range(matrix_size):
        m = int(interchanges[k])  
        b[k], b[m] = swap(b[k], b[m]) #interchange element k and m
    
    for j in range(matrix_size-1):
        for i in range(j+1, matrix_size):
            b[i] = b[i] - A[i][j]*b[j] #column oriented forward substition
            
    for j in reversed(range(0, matrix_size)):
        if A[j][j] == 0:
            print("Matrix A is singular\n")
            return

        b[j] = b[j]/A[j][j]
        for i in range(j):
            b[i] = b[i] - A[i][j]*b[j] #column oriented back substitution
    
if __name__ == "__main__":
    main()
    