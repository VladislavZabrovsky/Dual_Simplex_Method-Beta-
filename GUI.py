import tkinter as tk
from tkinter import ttk
from dual_simplex_OOP import DualSimplexMethod
import numpy as np


class SimplexCalculatorGUI:
    def __init__(self, root):
        """Initialization of GUI
           Here we can see organization of starting window
           User inputs Number of variables,constraints,objective
           After that user can push Submit button in order to
           come to next window"""
        self.root = root
        self.root.title("Simplex Calculator Input Window")

        self.num_variables_entry = ttk.Entry(root, width=10)
        self.num_variables_entry.grid(row=0, column=1, padx=10, pady=10)
        ttk.Label(root, text="Number of Variables:").grid(row=0, column=0)

        self.num_constraints_entry = ttk.Entry(root, width=10)
        self.num_constraints_entry.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(root, text="Number of Constraints:").grid(row=1, column=0)

        self.objective_var = tk.StringVar()
        self.objective_combobox = ttk.Combobox(root, textvariable=self.objective_var, values=["Maximize", "Minimize"])
        self.objective_combobox.grid(row=2, column=1, padx=10, pady=10)
        ttk.Label(root, text="Objective:").grid(row=2, column=0)

        ttk.Button(root, text="Submit", command=self.enter_coefficients).grid(row=3, column=0, columnspan=1, pady=5)

        self.coefficient_entries = []  # for objective function
        self.constraint_signs = []  # for signs
        self.constraint_entries = []  # for rhs_values
        self.constraint_matrix_entries = []  # for coefficients of constraint inequalities

        # Output labels for displaying parameters
        self.display_parameters_label = ttk.Label(root, text="")
        self.display_parameters_label.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(root, text="Solve", command=self.solve).grid(row=5, column=0, columnspan=1, pady=5)

    def update_parameters_display(self, num_variables, num_constraints, objective):
        """This method displays users input on a window"""
        parameters_text = f"Number of Variables: {num_variables}\nNumber of Constraints: {num_constraints}\nObjective: {objective}"

        if hasattr(self, 'display_parameters_label') and self.display_parameters_label.winfo_exists():
            self.display_parameters_label.config(text=parameters_text)
        else:
            self.display_parameters_label = ttk.Label(self.root, text=parameters_text)
            self.display_parameters_label.grid(row=3, column=2, padx=10, pady=10)

    def clear_text_on_screen(self):
        """Any time we press the Submit button script
        cleans the previous input and provides you with
        new one"""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Label) and widget.cget("text") not in ["Number of Variables:",
                                                                             "Number of Constraints:", "Objective:"]:
                widget.destroy()

    def enter_coefficients(self):
        """This method imitates the method  create_dual_simplex_input
           from  DualSimplexMethod class and helps us to extract
           arguments provided by user"""
        #Cleaning part
        self.clear_text_on_screen()

        for entry in self.coefficient_entries:
            entry.destroy()
        self.coefficient_entries = []

        for sign in self.constraint_signs:
            sign.destroy()
        self.constraint_signs = []


        for entry in self.constraint_entries:
            entry.destroy()
        self.constraint_entries = []

        for row_entries in self.constraint_matrix_entries:
            for entry in row_entries:
                entry.destroy()
        self.constraint_matrix_entries = []

        # Get user input from starting window
        num_variables = int(self.num_variables_entry.get())
        num_constraints = int(self.num_constraints_entry.get())
        objective = self.objective_var.get()

        # Update description on a screen
        self.update_parameters_display(num_variables, num_constraints, objective)

        # Create entry fields for coefficients
        for i in range(num_variables):
            entry = ttk.Entry(self.root, width=10)
            entry.grid(row=6, column=i + 1, padx=10, pady=10)
            self.coefficient_entries.append(entry)
            ttk.Label(self.root, text=f"Coefficient for x{i + 1}").grid(row=5, column=i + 1)

        for i in range(num_constraints):
            sign_var = tk.StringVar()
            sign_combobox = ttk.Combobox(self.root, textvariable=sign_var, values=[">=", "<="])
            sign_combobox.grid(row=i + 7, column=0, padx=10, pady=10)
            self.constraint_signs.append(sign_combobox)

            ttk.Label(self.root, text=f"Right-hand side of constraint {i + 1}").grid(row=i + 7,
                                                                                     column=num_variables + 1, padx=10,
                                                                                     pady=10)

            entry = ttk.Entry(self.root, width=10)
            entry.grid(row=i + 7, column=num_variables + 2, padx=10, pady=10)
            self.constraint_entries.append(entry)

            # Entry fields for coefficients of constraint inequalities
            constraint_matrix_entries_row = []
            for j in range(num_variables):
                matrix_entry = ttk.Entry(self.root, width=10)
                matrix_entry.grid(row=i + 7, column=j + 1, padx=10, pady=10)
                constraint_matrix_entries_row.append(matrix_entry)
            self.constraint_matrix_entries.append(constraint_matrix_entries_row)

    def solve(self):
        """This method incorporates next steps:
           1) Retrieval of all arguments from field
           2) Creating problem description on a screen
           3) Then we use already defined method in  DualSimplexMethod
              class in order to achieve the result """
        num_variables = int(self.num_variables_entry.get())
        num_constraints = int(self.num_constraints_entry.get())
        objective = "max" if self.objective_var.get() == "Maximize" else "min"

        # Retrieve coefficients from entry fields
        obj_coefficients = [float(entry.get()) for entry in self.coefficient_entries]
        constraint_coefficients = np.zeros((num_constraints, num_variables))
        constraint_signs = [sign_var.get() for sign_var in self.constraint_signs]
        rhs_values = [float(entry.get()) for entry in self.constraint_entries]

        for i in range(num_constraints):
            for j in range(num_variables):
                coefficient = float(self.constraint_matrix_entries[i][j].get())
                if constraint_signs[i] == ">=":
                    coefficient *= -1
                constraint_coefficients[i, j] = coefficient

        rhs_values = [rhs_values[i] * -1 if constraint_signs[i] == ">=" else rhs_values[i] for i in
                      range(num_constraints)]
        constraint_signs = ["<=" for _ in range(num_constraints)]

        dual_simplex_solver = DualSimplexMethod(num_variables, num_constraints, objective)
        dual_simplex_solver.obj_coefficients = np.array(obj_coefficients)
        dual_simplex_solver.A_matrix_parameters = constraint_coefficients
        dual_simplex_solver.B_matrix_parameters = np.array(rhs_values)
        dual_simplex_solver.signs = constraint_signs

        dual_simplex_solver.print_linear_programming_problem()
        table = dual_simplex_solver.create_simplex_table()
        dual_simplex_solver.print_simplex_table(table)
        dual_simplex_solver.dual_simplex_method()


def main():
    root = tk.Tk()
    app = SimplexCalculatorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
