import numpy as np

class Matrix:

    def __init__(self, matrix, convergence_threshold=1e-16, max_iterations=1000):
        self.n = matrix.shape[0]
        self.m = matrix.shape[1]
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.operate_records = []
        self.matrix = matrix
        if not self.is_hermi() or self.n != self.m:
            raise ValueError("Matrix must be hermitian.")

        self.matrix = matrix
        self.T = self.matrix.T
        self._tuple = tuple(map(tuple, self.matrix))

    def is_hermi(self):
        return np.allclose(self.matrix, self.matrix.conj().T)

    def copy(self):
        return Matrix(self.matrix.copy(), self.convergence_threshold, self.max_iterations)

    def exchange_rows(self, i, j):
        self.matrix[[i, j], :] = self.matrix[[j, i], :]

    def exchange_columns(self, i, j):
        self.matrix[:, [i, j]] = self.matrix[:, [j, i]]


class Unitary4Eigen:

    def __init__(self, theta, dimension, target_2rows_index: tuple, target_2columns_index: tuple):
        self.theta = theta
        self.target_2rows_index = target_2rows_index
        self.target_2columns_index = target_2columns_index
        self.matrix2D = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        self.matrix = np.identity(dimension)
        self.matrix[np.ix_(target_2rows_index, target_2columns_index)] = self.matrix2D

    def get_operation_record(self):
        return (self.target_2rows_index) , self.theta    


class Iterator:
    def __init__(self, input: Matrix):
        self.M = input

    def pending_operations(self):
        max_r = self.M.m
        _continue = True
        while _continue:
            count = 0
            for r in range(max_r):
                for c in range(r):
                    print(f"Checking element at ({r}, {c}): {self.M.matrix[r, c]}")
                    if abs(float(self.M.matrix[r, c])) > self.M.convergence_threshold:
                        print(f"Pending operation found at ({r}, {c}) with value {self.M.matrix[r, c]}")
                        count += 1
                        yield r, c
            if count == 0:
                _continue = False
                print("Converged")
                return

    def iterate(self):
        for r, c in self.pending_operations():
            theta = 0.5 * np.arctan2(2 * self.M.matrix[r, c], self.M.matrix[c, c] - self.M.matrix[r, r])
            U = Unitary4Eigen(theta, self.M.n, (c, r), (c, r))
            self.M.matrix = U.matrix @ self.M.matrix @ U.matrix.T
            self.M.operate_records.append(U.get_operation_record())


def __main__():
    A = np.array([[4, 1, 2], [1, 3, 0], [2, 0, 1]])
    matrix = Matrix(A)
    iterator = Iterator(matrix)
    iterator.iterate()
    print("Eigenvalues:", np.diag(matrix.matrix))
    print("Operation Records:", matrix.operate_records)
    print("Final Matrix:\n", matrix.matrix)

if __name__ == "__main__":
    __main__()