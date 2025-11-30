"""
A proof-of-concept Python module that mimics the 'gurobipy' API
and uses Google OR-Tools as the backend solver.

This is a *demonstration* and not a complete, production-ready
drop-in replacement. Many features, attributes, and parameters
are not implemented.

To use this, you must have Google OR-Tools installed:
'pip install ortools'
"""

from ortools.linear_solver import pywraplp


class GRB:
    """
    A class to hold constants mimicking Gurobi's GRB object.
    """
    # Optimization status
    OPTIMAL = pywraplp.Solver.OPTIMAL
    INFEASIBLE = pywraplp.Solver.INFEASIBLE
    UNBOUNDED = pywraplp.Solver.UNBOUNDED
    FEASIBLE = pywraplp.Solver.FEASIBLE
    NOT_SOLVED = pywraplp.Solver.NOT_SOLVED

    # Variable types
    CONTINUOUS = pywraplp.Solver.NumVar
    BINARY = pywraplp.Solver.BoolVar
    INTEGER = pywraplp.Solver.IntVar

    # Senses
    MINIMIZE = pywraplp.Solver.Minimize
    MAXIMIZE = pywraplp.Solver.Maximize

    # A default "infinity" value
    INFINITY = pywraplp.Solver.infinity()


# --- Custom Exception Class ---
class OpenGurobiError(Exception):
    """
    A custom exception for errors in this wrapper.
    """
    pass


# --- Wrapper for 'Var' class ---
class Var:
    """
    A wrapper for an OR-Tools variable, mimicking gurobipy.Var.
    """
    def __init__(self, ortools_var, model_wrapper):
        self._var = ortools_var
        self._model = model_wrapper
        self.VarName = ortools_var.name()

        # Attributes that get populated after solving
        self.X = None  # Solution value
        self.RC = None  # Reduced cost

    def __str__(self):
        return f"<OpenGurobi Var {self.VarName}>"

    def __repr__(self):
        return self.__str__()

    # --- Overload operators to build linear expressions ---
    def __add__(self, other):
        return LinExpr(self) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return LinExpr(self) - other

    def __rsub__(self, other):
        # other - self
        return other + (self * -1)

    def __mul__(self, other):
        return LinExpr(self) * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1


# --- Wrapper for 'LinExpr' class ---
class LinExpr:
    """
    A class to build linear expressions, mimicking gurobipy.LinExpr.
    This is necessary because OR-Tools builds expressions differently.
    """
    def __init__(self, initial_expr=None, constant=0.0):
        self.terms = {}  # Dictionary of {Var: coeff}
        self.constant = constant

        if isinstance(initial_expr, Var):
            self.terms[initial_expr] = 1.0
        elif isinstance(initial_expr, (int, float)):
            self.constant = float(initial_expr)
        elif initial_expr is not None:
            raise OpenGurobiError(
                f"Cannot initialize LinExpr with type {type(initial_expr)}"
            )

    def add(self, other, coeff=1.0):
        """Adds another expression or variable."""
        if isinstance(other, Var):
            self.terms[other] = self.terms.get(other, 0.0) + coeff
        elif isinstance(other, LinExpr):
            for var, c in other.terms.items():
                self.terms[var] = self.terms.get(var, 0.0) + c * coeff
            self.constant += other.constant * coeff
        elif isinstance(other, (int, float)):
            self.constant += float(other) * coeff
        else:
            raise OpenGurobiError(f"Cannot add type {type(other)} to LinExpr")

    def __add__(self, other):
        new_expr = LinExpr()
        new_expr.add(self)
        new_expr.add(other)
        return new_expr

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new_expr = LinExpr()
        new_expr.add(self)
        new_expr.add(other, coeff=-1.0)
        return new_expr

    def __rsub__(self, other):
        # other - self
        new_expr = LinExpr(other)
        new_expr.add(self, coeff=-1.0)
        return new_expr

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise OpenGurobiError(
                "Can only multiply LinExpr by a constant (no quadratic)"
            )

        new_expr = LinExpr()
        for var, coeff in self.terms.items():
            new_expr.terms[var] = coeff * other
        new_expr.constant = self.constant * other
        return new_expr

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    # --- Overload comparison operators to create constraints ---
    def __le__(self, other):
        # self <= other  =>  self - other <= 0
        expr = self - other
        return (expr, '<=', 0)  # Returns a tuple representing the constraint

    def __ge__(self, other):
        # self >= other  =>  self - other >= 0
        expr = self - other
        return (expr, '>=', 0)  # Returns a tuple representing the constraint

    def equals(self, other):
        # self == other  =>  self - other == 0
        expr = self - other
        return (expr, '==', 0)  # Returns a tuple representing the constraint

    def __eq__(self, other) -> bool:
        # Preserve standard equality behavior for type checkers and Python
        return object.__eq__(self, other)


# --- The main 'Model' class ---
class Model:
    """
    A wrapper for an OR-Tools Solver, mimicking gurobipy.Model.
    """
    def __init__(self, name=""):
        # We choose the 'CBC' solver here, which is a strong open-source
        # MILP solver. You could also use 'SCIP' or 'GLOP'.
        # For multi-threading, 'SCIP' or the CP-SAT solver are better choices.
        # This example uses the 'CBC' (COIN-OR Branch and Cut) solver.
        self.solver = pywraplp.Solver.CreateSolver('CBC')
        if not self.solver:
            raise OpenGurobiError("Could not create CBC solver instance")

        self.name = name
        self._vars = {}  # {name: Var}
        self.Status = GRB.NOT_SOLVED
        self.ObjVal = None

    def addVar(
        self,
        lb=0.0,
        ub=GRB.INFINITY,
        obj=0.0,
        vtype=GRB.CONTINUOUS,
        name="",
    ):
        """
        Mimics the gurobipy.Model.addVar() method.
        """
        if name in self._vars:
            raise OpenGurobiError(f"Variable '{name}' already exists")

        if vtype == GRB.CONTINUOUS:
            ort_var = self.solver.NumVar(lb, ub, name)
        elif vtype == GRB.BINARY:
            ort_var = self.solver.BoolVar(name)
        elif vtype == GRB.INTEGER:
            ort_var = self.solver.IntVar(lb, ub, name)
        else:
            raise OpenGurobiError(f"Unknown variable type {vtype}")

        # Add to objective if 'obj' coefficient is provided
        if obj != 0.0:
            self.solver.Objective().SetCoefficient(ort_var, obj)

        var_wrapper = Var(ort_var, self)
        self._vars[name] = var_wrapper
        return var_wrapper

    def setObjective(self, expr, sense=GRB.MINIMIZE):
        """
        Mimics the gurobipy.Model.setObjective() method.
        """
        if not isinstance(expr, LinExpr):
            raise OpenGurobiError("Objective must be a LinExpr")

        objective = self.solver.Objective()
        # Clear existing objective terms
        objective.Clear()

        for var, coeff in expr.terms.items():
            objective.SetCoefficient(var._var, coeff)

        objective.SetOffset(expr.constant)

        if sense == GRB.MINIMIZE:
            objective.SetMinimization()
        elif sense == GRB.MAXIMIZE:
            objective.SetMaximization()
        else:
            raise OpenGurobiError(f"Unknown objective sense {sense}")

    def addConstr(self, constraint_tuple, name=""):
        """
        Mimics the gurobipy.Model.addConstr() method.
        This version expects a tuple from an overloaded comparison,
        e.g., (x + y <= 1)
        """
        if not (
            isinstance(constraint_tuple, tuple) and len(constraint_tuple) == 3
        ):
            raise OpenGurobiError(
                "Invalid constraint. Use 'model.addConstr(x + y <= 1)'"
            )

        expr, sense, rhs = constraint_tuple

        # We need to build the OR-Tools constraint.
        # The form is: lb <= [expression] <= ub

        # Our expression is (expr - rhs)
        # So we have (expr.terms) and (expr.constant - rhs)

        ort_expr = 0.0
        for var, coeff in expr.terms.items():
            ort_expr += var._var * coeff

        constant = expr.constant

        # Now create the constraint based on the sense
        # OR-Tools constraints are of the form:
        # solver.Add(expression, lb, ub, name)

        if sense == '<=':
            # expr <= rhs  =>  ort_expr + constant <= rhs
            # hence ort_expr <= rhs - constant
            ub = rhs - constant
            lb = -GRB.INFINITY
            ort_constr = self.solver.Add(ort_expr <= ub, name)
        elif sense == '>=':
            # expr >= rhs  =>  ort_expr + constant >= rhs
            # hence ort_expr >= rhs - constant
            lb = rhs - constant
            ub = GRB.INFINITY
            ort_constr = self.solver.Add(ort_expr >= lb, name)
        elif sense == '==':
            # expr == rhs  =>  ort_expr + constant == rhs
            # hence ort_expr == rhs - constant
            val = rhs - constant
            ort_constr = self.solver.Add(ort_expr == val, name)
        else:
            raise OpenGurobiError(f"Unknown constraint sense {sense}")

        # We would return a 'Constr' wrapper object here,
        # but we'll skip that for this simple example.
        return ort_constr

    def optimize(self):
        """
        Mimics the gurobipy.Model.optimize() method.
        """

        # --- This is where you would set parallel computing options ---
        # For example, if using SCIP or a solver that supports it:
        # self.solver.SetSolverSpecificParametersAsString("threads=8")
        # For CBC, multi-threading is more complex.
        # For OR-Tools CP-SAT, it's parallel by default.
        # This example uses the standard CBC solver, which is single-threaded.

        print(f"Starting optimization with {self.solver.SolverVersion()}...")
        status = self.solver.Solve()

        # --- Update model and variable attributes ---
        self.Status = status

        if status == GRB.OPTIMAL:
            self.ObjVal = self.solver.Objective().Value()
            print(f"Optimal solution found. Objective: {self.ObjVal}")
            for name, var in self._vars.items():
                var.X = var._var.solution_value()
                try:
                    var.RC = var._var.reduced_cost()
                except Exception as e:
                    print(f"Could not get reduced cost for '{name}': {e}")
                    var.RC = None   # Not always available

        elif status == GRB.FEASIBLE:
            self.ObjVal = self.solver.Objective().Value()
            print(f"Feasible solution found. Objective: {self.ObjVal}")
            for name, var in self._vars.items():
                var.X = var._var.solution_value()

        elif status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print("Solver did not find an optimal solution.")

    def getVarByName(self, name):
        """
        Mimics the gurobipy.Model.getVarByName() method.
        """
        if name not in self._vars:
            raise OpenGurobiError(f"Variable '{name}' not found.")
        return self._vars[name]

    def setParam(self, param_name, value):
        """
        A simple mimic of setParam.
        This is HIGHLY solver-specific.
        """
        print(f"setParam('{param_name}', {value}) is not fully implemented.")
        if param_name.lower() == "timelimit":
            self.solver.set_time_limit(int(value * 1000))  # OR-Tools uses ms
        if param_name.lower() == "threads":
            # This is tricky. For CBC, this isn't a simple parameter.
            # For SCIP, it would be:
            # self.solver.SetSolverSpecificParametersAsString(f"lp/threads={value}")
            pass


# --- Example Usage ---
if __name__ == "__main__":

    print("--- Testing Open-Source Gurobi-like API ---")

    # Create a new model
    m = Model("my_milp_model")

    # --- Create variables ---
    # m.addVar(lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="")
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.INTEGER, lb=0, ub=10, name="z")

    # --- Set objective ---
    # We want to maximize: 1*x + 1*y + 2*z
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # --- Add constraints ---
    # c1: x + 2*y + 3*z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c1")

    # c2: x + y >= 1
    m.addConstr(x + y >= 1, "c2")

    # c3: z == 1
    # m.addConstr(z == 1, "c3")

    # --- Set parameters ---
    m.setParam("TimeLimit", 60)  # 60 seconds
    m.setParam("Threads", 8)   # This is a mock-up for this example

    # --- Optimize the model ---
    m.optimize()

    # --- Print results ---
    if m.Status == GRB.OPTIMAL:
        print("\n--- Solution ---")
        print(f"Objective Value: {m.ObjVal}")
        for v in m._vars.values():
            print(f"{v.VarName} = {v.X}")

        # You can also access variables by name
        print("\nAccessing 'x' by name:")
        x_var = m.getVarByName("x")
        print(f"x = {x_var.X}")
