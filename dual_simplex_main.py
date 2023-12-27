from dual_simplex_OOP import DualSimplexMethod

def main():
    try:
        num_variables_input = input("Enter the number of variables: ")
        num_constraints_input = input("Enter the number of constraints: ")

        if not num_variables_input.isdigit() or not num_constraints_input.isdigit():
            raise ValueError("Number of variables and constraints must be positive integers.")

        num_variables = int(num_variables_input)
        num_constraints = int(num_constraints_input)

        if num_variables <= 0 or num_constraints <= 0:
            raise ValueError("Number of variables and constraints must be positive integers.")

        objective = input("Enter 'max'/'min' for Maximization/Minimization: ").lower()

        if objective not in ['min', 'max']:
            raise ValueError("Objective must be 'min' or 'max'.")

        dual_simplex_solver = DualSimplexMethod(num_variables, num_constraints, objective)
        dual_simplex_solver.create_dual_simplex_input()
        dual_simplex_solver.print_linear_programming_problem()
        table = dual_simplex_solver.create_simplex_table()
        dual_simplex_solver.print_simplex_table(table)
        dual_simplex_solver.dual_simplex_method()

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

