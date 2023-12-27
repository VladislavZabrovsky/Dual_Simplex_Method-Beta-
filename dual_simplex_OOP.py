import numpy as np
import pandas as pd


class DualSimplexMethod:
    def __init__(self, num_variables, num_constraints,objective):
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.obj_coefficients = None
        self.A_matrix_parameters = None
        self.B_matrix_parameters = None
        self.signs = None
        self.objective = objective

    def create_dual_simplex_input(self):
        """Description:
               num_variables: how many variables are present
               num_constraints: how many constraint equetions are present
               Function serves for taking input of user,
               thus initializes A matrix,b vector and coefficients
               of objective function:
               Ax = b
               Max Z = c1x1 + .....cnxn - objective function for n variables
            """

        self.obj_coefficients = np.array(
            [float(input(f'Enter coefficient for variable x{i + 1} in the objective function: ')) for i in
             range(self.num_variables)])

        self.A_matrix_parameters = np.zeros((self.num_constraints, self.num_variables))

        self.B_matrix_parameters = np.zeros(self.num_constraints)

        self.signs = []
        for i in range(self.num_constraints):
            sign = input(f"\nEnter the sign of inequality('>=','<=') for constraint {i + 1}: ")
            if sign not in ['>=', '<=']:
                raise ValueError("Inappropriate sign for given problem")
            else:
                self.signs.append(sign.strip())
            print(f'Enter coefficients for constraint {i + 1}:')
            for j in range(self.num_variables):
                self.A_matrix_parameters[i, j] = float(input(f' Coefficient for variable x{j + 1}: '))
                if self.signs[i] == '>=':
                    self.A_matrix_parameters[i, j] *= -1
            self.B_matrix_parameters[i] = float(input('Enter the constraints from right side: '))
            if self.signs[i] == '>=':
                self.B_matrix_parameters[i] *= -1
            self.signs[i] = "<="
        return self.obj_coefficients,  self.A_matrix_parameters,  self.B_matrix_parameters,  self.signs

    def print_linear_programming_problem(self):
        """Description:
               Function serves for printing a linear problem
               to solve
               Parameters:
               obj_coefficients - coefficients of objective function
               constraint_coefficients - values of A matrix in Ax = b
               rhs_values - values of b vector in Ax = b
               objective - Minimization or Maximization problem
               Example:
                   Objective Function:
                   Maximize Z = 2.0x1 + -1.0x2

                    System to solve:
                    2.0x1 + 1.0x2 <= 18.0
                    1.0x1 + 3.0x2 <= 12.0
                    3.0x1 + -8.0x2 <= 16.0
                    x1,x2 >= 0
            """
        num_variables = len(self.obj_coefficients)
        num_constraints = len(self.B_matrix_parameters)

        if self.objective == 'max':
            objective_state = "Maximize"
        elif self.objective == 'min':
            objective_state = "Minimize"
        else:
            raise ValueError("Inappropriate problem to solve")

        print("\nObjective Function:")

        objective_function = f"{objective_state} Z = {self.obj_coefficients[0]}x1"
        for i in range(1, num_variables):
            objective_function += f" + {self.obj_coefficients[i]}x{i + 1}"
        print(objective_function)

        print("\nSystem to solve:")
        for i in range(num_constraints):
            constraint = f"{self.A_matrix_parameters[i, 0]}x1"
            for j in range(1, num_variables):
                constraint += f" + {self.A_matrix_parameters[i, j]}x{j + 1}"
            constraint += f" {self.signs[i]} {self.B_matrix_parameters[i]}"
            print(constraint)
        print(','.join(f"x{k + 1}" for k in range(num_variables)), end="")
        print(' >= 0')
        print(" ")



    def create_simplex_table(self):
        """Description:
               Function serves for creation of simplex table
               Parameters:
               obj_coefficients - coefficients of objective function
               constraint_coefficients - values of A matrix in Ax = b
               rhs_values - values of b vector in Ax = b
               """
        num_variables = len(self.obj_coefficients)
        num_constraints = len(self.B_matrix_parameters)

        # Create an identity matrix for the basis columns
        basis_addition = np.eye(num_constraints)

        # Augment the coefficients for the tableau
        augmented_matrix = np.hstack((self.B_matrix_parameters.reshape(-1, 1), self.A_matrix_parameters, basis_addition))

        # Create Z_line for the objective function coefficients
        if self.objective == "max":
            Z_line = np.hstack(
                (np.zeros(1), self.obj_coefficients * -1,
                 np.zeros(augmented_matrix.shape[1] - self.obj_coefficients.shape[0] - 1)))
        else:
            Z_line = np.hstack(
                (np.zeros(1), self.obj_coefficients, np.zeros(augmented_matrix.shape[1] - self.obj_coefficients.shape[0] - 1)))

        # Stack the augmented_matrix and Z_line to form the tableau coefficients
        coefficients_for_table = np.vstack((augmented_matrix, Z_line))

        # Create a DataFrame from the coefficients
        df = pd.DataFrame(coefficients_for_table,
                          columns=['b'] + [f'x{i + 1}' for i in range(num_variables + num_constraints)])

        # Set 'Basis' as the index
        df.insert(0, 'Basis', [f'x{i + num_variables + 1}' for i in range(num_constraints)] + ['Z'])
        df.set_index("Basis", inplace=True)

        return df

    def print_simplex_table(self,simplex_table):
        """Description:
               Prints Simplex tables"""
        print("SIMPLEX METHOD TABLE")
        print(simplex_table)

    def perform_gaussian_elimination(self,df,pivot_row,pivot_column):
        """Jordan Gauss method for recalculation of coefficients"""
        for basis in df.index:
            if basis != pivot_row:
                multiplier = df.at[basis, pivot_column]
                df.loc[basis] -= multiplier * df.loc[pivot_row]
        return df

    def dual_simplex_method(self):
        """Description:
               Implemented dual simplex algorithm
            """
        try:
            df = self.create_simplex_table()

            iterations = 0

            while np.any(df["b"].drop('Z') < 0):
                pivot_row = df["b"].drop("Z").idxmin()
                negative_coefficients = df.loc[pivot_row, df.loc[pivot_row] < 0].drop('b')
                obj_coef_negative = df.loc["Z", df.loc[pivot_row] < 0].drop('b')
                pivot_column = np.abs(obj_coef_negative / negative_coefficients).idxmin()
                main_element = df.at[pivot_row, pivot_column]
                df.loc[pivot_row] /= main_element
                df = self.perform_gaussian_elimination(df, pivot_row, pivot_column)
                df = df.rename(index={pivot_row: pivot_column})

                iterations += 1
                print(f"\nIteration: {iterations}")
                self.print_simplex_table(df)
                print(f"Pivot_row: {pivot_row}")
                print(f"Pivot_column: {pivot_column}")
                print(f"Main_element: {main_element}")
                print(" " * 20)

            while np.min(df.loc["Z"][1:]) < 0:
                valid_columns = df.columns[1:]  # we don't include b column
                pivot_column = df.loc["Z", valid_columns].idxmin()
                ratio = df['b'].drop('Z') / df[pivot_column].drop('Z')
                ratio = ratio[ratio > 0]
                pivot_row = ratio.idxmin()

                main_element = df.at[pivot_row, pivot_column]
                df.loc[pivot_row] /= main_element
                df = self.perform_gaussian_elimination(df, pivot_row, pivot_column)
                df = df.rename(index={pivot_row: pivot_column})

                iterations += 1
                print(f"Iteration: {iterations}")
                self.print_simplex_table(df)
                print(f"Pivot_column: {pivot_column} ")
                print(f"Pivot_row: {pivot_row} ")
                print(f"Main_element: {main_element}")
                print(" " * 20)

            optimal_z = df.at['Z', 'b']

            if self.objective == "min":
                optimal_z *= -1

            main_variables = [f'x{i}' for i in range(1, len(self.obj_coefficients) + 1)]

            optimal_vector = {var: 0 for var in main_variables}

            for var in df.index:
                if var in optimal_vector:
                    optimal_vector[var] = df.at[var, 'b']

            print(f"Optimal Z value: {optimal_z}")
            print(f"Optimal Vector: {optimal_vector}")

        except Exception:
            print("Problem doesn't have a solution or has multiple solutions")


