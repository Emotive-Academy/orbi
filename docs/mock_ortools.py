import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# Mock Constants
class Solver:
    OPTIMAL = 0
    FEASIBLE = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    ABNORMAL = 4
    NOT_SOLVED = 6

    NumVar = 0
    IntVar = 1
    BoolVar = 2

    Minimize = 0
    Maximize = 1

    @staticmethod
    def infinity():
        return float('inf')

    @staticmethod
    def CreateSolver(name):
        return SolverInstance(name)


class Variable:
    def __init__(self, name, lb, ub, vtype):
        self._name = name
        self._lb = lb
        self._ub = ub
        self._vtype = vtype
        self._solution_value = 0.0

    def name(self):
        return self._name

    def solution_value(self):
        return self._solution_value

    def reduced_cost(self):
        return 0.0  # type: ignore  Not implemented in SciPy MILP easily


class Objective:
    def __init__(self):
        self._coeffs = {}  # var -> coeff
        self._offset = 0.0
        self._direction = Solver.Minimize

    def SetCoefficient(self, var, coeff):
        self._coeffs[var] = coeff

    def SetOffset(self, value):
        self._offset = value

    def SetMinimization(self):
        self._direction = Solver.Minimize

    def SetMaximization(self):
        self._direction = Solver.Maximize

    def Clear(self):
        self._coeffs = {}
        self._offset = 0.0

    def Value(self):
        # This is expected to be called after solve
        # We need the solver to inject the value or calculate it
        # But commonly we just return the value computed by solver
        return self._value  # type: ignore


class SolverInstance:
    def __init__(self, name):
        self._variables = []
        self._constraints = []
        self._objective = Objective()
        self._timelimit = float('inf')

    def SolverVersion(self):
        return "SciPy MILP (Browser Mock)"

    def NumVar(self, lb, ub, name):
        v = Variable(name, lb, ub, Solver.NumVar)
        self._variables.append(v)
        return v

    def IntVar(self, lb, ub, name):
        v = Variable(name, lb, ub, Solver.IntVar)
        self._variables.append(v)
        return v

    def BoolVar(self, name):
        v = Variable(name, 0, 1, Solver.IntVar)  # Bool is Int 0..1
        self._variables.append(v)
        return v

    def Objective(self):
        return self._objective

    def Add(self, constraint_expr, name=""):
        # This is tricky because orbi passes `ort_expr <= ub` which is an
        # object in OR-Tools
        # In orbi/milp.py, `ort_expr` corresponds to `var * coeff + ...`
        # But `orbi` uses `ort_expr <= ub` which calls operator overloading on
        # the OR-Tools objects.
        # Oh wait! In `orbi/milp.py`, it does:
        # ort_constr = self.solver.Add(ort_expr <= ub, name)
        #
        # But `ort_expr` is built using `var._var * coeff`.
        # `var._var` is OUR Mock Variable.        # So we need to support
        # operator overloading on OUR Mock
        # Variable/Expression to return something `solver.Add` can consume.

        # Actually, `orbi` wraps everything in its OWN `LinExpr`.
        # When it calls `self.solver.Add(ort_expr <= ub)`, `ort_expr` is a
        # Python sum of `ortools_var * coeff`.
        # So our Mock Variable needs to support `__mul__`, `__add__` to build
        # a MockExpression.
        # And MockExpression needs `__le__`, `__ge__`, `__eq__` to return a
        # Constraint object.
        pass  # implemented dynamically below by patching Variable
        return None

    def Solve(self):
        n_vars = len(self._variables)
        if n_vars == 0:
            return Solver.OPTIMAL  # trivial

        # 1. Prepare Integrality
        # 0 = continuous, 1 = integer. Bool is integer.
        integrality = np.zeros(n_vars)
        for i, v in enumerate(self._variables):
            if v._vtype == Solver.IntVar or v._vtype == Solver.BoolVar:
                integrality[i] = 1

        # 2. Prepare Bounds
        lbs = [v._lb for v in self._variables]
        ubs = [v._ub for v in self._variables]
        bounds = Bounds(lbs, ubs)  # type: ignore

        # 3. Prepare Objective
        # SciPy minimizes c @ x. If maximizing, we negate c.
        c = np.zeros(n_vars)
        for var, coeff in self._objective._coeffs.items():
            idx = self._variables.index(var)
            c[idx] = coeff

        if self._objective._direction == Solver.Maximize:
            c = -c

        # 4. Prepare Constraints
        # We need A_ub @ x <= b_ub and A_eq @ x == b_eq
        # But SciPy's LinearConstraint uses lb <= A @ x <= ub
        # This is flexible.

        A_rows = []
        c_lbs = []
        c_ubs = []

        for c_obj in self._constraints:
            # c_obj is (coeffs_dict, lb, ub)
            row = np.zeros(n_vars)
            for var, coeff in c_obj['coeffs'].items():
                idx = self._variables.index(var)
                row[idx] = coeff
            A_rows.append(row)
            c_lbs.append(c_obj['lb'])
            c_ubs.append(c_obj['ub'])

        constraints_arg = None
        if A_rows:
            A = np.vstack(A_rows)
            constraints_arg = LinearConstraint(A, c_lbs, c_ubs)  # type: ignore

        # 5. Run SciPy MILP
        try:
            res = milp(c=c, constraints=constraints_arg,
                       integrality=integrality, bounds=bounds)
        except Exception as e:
            print(f"SciPy Error: {e}")
            return Solver.ABNORMAL

        # 6. Map results back
        if res.success:
            for i, v in enumerate(self._variables):
                v._solution_value = res.x[i]

            # Set objective value
            # Remember to un-negate if maximization
            obj_val = (c @ res.x)
            if self._objective._direction == Solver.Maximize:
                obj_val = -obj_val

            # Add offset
            obj_val += self._objective._offset
            self._objective._value = obj_val  # type: ignore

            return Solver.OPTIMAL
        else:
            if "infeasible" in res.message.lower():
                return Solver.INFEASIBLE
            if "unbounded" in res.message.lower():
                return Solver.UNBOUNDED
            return Solver.NOT_SOLVED

    def set_time_limit(self, ms):
        self._timelimit = ms


# --- Helper Classes for Expression Building ---
class LinearExpr:
    def __init__(self, coeffs=None, constant=0.0):
        self.coeffs = coeffs if coeffs else {}  # var -> coeff
        self.constant = constant

    def __add__(self, other):
        new_coeffs = self.coeffs.copy()
        new_const = self.constant

        if isinstance(other, LinearExpr):
            for v, c in other.coeffs.items():
                new_coeffs[v] = new_coeffs.get(v, 0.0) + c
            new_const += other.constant
        elif isinstance(other, Variable):
            new_coeffs[other] = new_coeffs.get(other, 0.0) + 1.0
        elif isinstance(other, (int, float)):
            new_const += other
        return LinearExpr(new_coeffs, new_const)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {v: c * other for v, c in self.coeffs.items()}
            return LinearExpr(new_coeffs, self.constant * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __le__(self, other):
        # expr <= other. Convert to lb <= expr <= ub
        # self.coeffs * vars + self.constant <= other
        # self.coeffs * vars <= other - self.constant
        # -inf <= ... <= other - self.constant
        val = other
        if isinstance(other, LinearExpr):
            # This case shouldn't strictly happen in basic usage of orbi but
            # possible
            # Simplified: assuming other is constant as orbi handles
            # expression algebra
            pass

        return {'coeffs': self.coeffs, 'lb': -float('inf'), 'ub': float(val) - self.constant}  # flake8: E501

    def __ge__(self, other):
        # expr >= other
        # self.coeffs * vars >= other - self.constant
        return {'coeffs': self.coeffs, 'lb': float(other) - self.constant, 'ub': float('inf')}  # flake8: E501

    def __eq__(self, other):
        # expr == other
        return {'coeffs': self.coeffs, 'lb': float(other) - self.constant,
                'ub': float(other) - self.constant}  # flake8: E501


# Patch Variable to support arithmetic
def var_add(self, other):
    return LinearExpr({self: 1.0}) + other


def var_sub(self, other):
    return LinearExpr({self: 1.0}) - other


def var_mul(self, other):
    return LinearExpr({self: 1.0}) * other


def var_le(self, other):
    return LinearExpr({self: 1.0}) <= other


def var_ge(self, other):
    return LinearExpr({self: 1.0}) >= other


def var_eq(self, other):
    return LinearExpr({self: 1.0}) == other


Variable.__add__ = var_add  # type: ignore
Variable.__radd__ = var_add  # type: ignore
Variable.__sub__ = var_sub  # type: ignore
Variable.__rsub__ = lambda self, other: LinearExpr(constant=other) - LinearExpr({self: 1.0})   # flake8: E501
Variable.__mul__ = var_mul  # type: ignore
Variable.__rmul__ = var_mul  # type: ignore
Variable.__le__ = var_le  # type: ignore
Variable.__ge__ = var_ge  # type: ignore
Variable.__eq__ = var_eq


# Patch SolverInstance.Add
def solver_add(self, constraint_dict, name=""):
    # Constraint dict came from LinearExpr comparison
    self._constraints.append(constraint_dict)
    return None  # Return mock constraint if needed


SolverInstance.Add = solver_add

# define pywraplp
pywraplp = type('pywraplp', (), {})()
pywraplp.Solver = Solver
