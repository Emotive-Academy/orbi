
document.addEventListener('DOMContentLoaded', async () => {
    const runBtn = document.getElementById('run-btn');
    const outputDiv = document.getElementById('output');
    const codeBlock = document.getElementById('code-block');
    const exampleSelect = document.getElementById('example-select');
    const loadingIndicator = document.getElementById('loading-indicator');

    let pyodide = null;
    let pyodideReady = false;

    // --- initialization ---
    async function initPyodideEnv() {
        try {
            console.log("Loading Pyodide...");
            pyodide = await loadPyodide();

            loadingIndicator.innerText = "Installing dependencies (numpy, scipy)...";
            await pyodide.loadPackage("numpy");
            await pyodide.loadPackage("scipy");

            loadingIndicator.innerText = "Loading orbi library...";

            // 1. Inline mock_ortools.py content to avoid fetch errors on local file protocol
            const mockOrtoolsCode = `
import sys
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
        return 0.0

class Objective:
    def __init__(self):
        self._coeffs = {}
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
        return self._value

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
        v = Variable(name, 0, 1, Solver.IntVar)
        self._variables.append(v)
        return v
    def Objective(self):
        return self._objective
    def Add(self, constraint_expr, name=""):
        if constraint_expr is not None:
             self._constraints.append(constraint_expr)
        return None
    def Solve(self):
        n_vars = len(self._variables)
        if n_vars == 0:
            return Solver.OPTIMAL

        # Create map for O(1) loop up and identity based check
        var_map = {v: i for i, v in enumerate(self._variables)}

        integrality = np.zeros(n_vars)
        for i, v in enumerate(self._variables):
            if v._vtype == Solver.IntVar or v._vtype == Solver.BoolVar:
                integrality[i] = 1
        lbs = [v._lb for v in self._variables]
        ubs = [v._ub for v in self._variables]
        bounds = Bounds(lbs, ubs)

        # Objective
        c = np.zeros(n_vars)
        for var, coeff in self._objective._coeffs.items():
            if var in var_map:
                c[var_map[var]] = coeff
        if self._objective._direction == Solver.Maximize:
            c = -c

        # Constraints
        A_rows = []
        c_lbs = []
        c_ubs = []
        for c_obj in self._constraints:
            row = np.zeros(n_vars)
            if isinstance(c_obj, dict) and 'coeffs' in c_obj:
                for var, coeff in c_obj['coeffs'].items():
                    if var in var_map:
                        row[var_map[var]] = coeff
                A_rows.append(row)
                c_lbs.append(c_obj['lb'])
                c_ubs.append(c_obj['ub'])

        constraints_arg = None
        if A_rows:
            A = np.vstack(A_rows)
            constraints_arg = LinearConstraint(A, c_lbs, c_ubs)
        try:
            res = milp(c=c, constraints=constraints_arg, integrality=integrality, bounds=bounds)
        except Exception as e:
            print(f"SciPy Error: {e}")
            return Solver.ABNORMAL
        if res.success:
            for i, v in enumerate(self._variables):
                v._solution_value = res.x[i]
            obj_val = (c @ res.x)
            if self._objective._direction == Solver.Maximize:
                obj_val = -obj_val
            obj_val += self._objective._offset
            self._objective._value = obj_val
            return Solver.OPTIMAL
        else:
            if "infeasible" in res.message.lower():
                return Solver.INFEASIBLE
            if "unbounded" in res.message.lower():
                return Solver.UNBOUNDED
            return Solver.NOT_SOLVED
    def set_time_limit(self, ms):
        self._timelimit = ms

class LinearExpr:
    def __init__(self, coeffs=None, constant=0.0):
        self.coeffs = coeffs if coeffs else {}
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

    # Comparison operators return constraint objects (dicts)
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return {'coeffs': self.coeffs, 'lb': -float('inf'), 'ub': float(other) - self.constant}
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, (int, float)):
             return {'coeffs': self.coeffs, 'lb': float(other) - self.constant, 'ub': float('inf')}
        return NotImplemented
    def __eq__(self, other):
        if isinstance(other, (int, float)):
             return {'coeffs': self.coeffs, 'lb': float(other) - self.constant, 'ub': float(other) - self.constant}
        # Force default object equality if comparing to non-numbers
        return object.__eq__(self, other)

# Patch Variable to support arithmetic but NOT comparisons (to preserve hash/eq)
def var_add(self, other):
    return LinearExpr({self: 1.0}) + other
def var_sub(self, other):
    return LinearExpr({self: 1.0}) - other
def var_mul(self, other):
    return LinearExpr({self: 1.0}) * other

Variable.__add__ = var_add
Variable.__radd__ = var_add
Variable.__sub__ = var_sub
Variable.__rsub__ = lambda self, other: LinearExpr(constant=other) - LinearExpr({self: 1.0})
Variable.__mul__ = var_mul
Variable.__rmul__ = var_mul

# We do NOT patch __eq__, __le__, __ge__ on Variable.
# Users must wrap in LinearExpr or arithmetic to generate constraints,
# which matches orbi behavior (ort_expr is LinearExpr).
def solver_add(self, constraint_dict, name=""):
    if constraint_dict is not None:
        self._constraints.append(constraint_dict)
    return None
SolverInstance.Add = solver_add
pywraplp = type('pywraplp', (), {})()
pywraplp.Solver = Solver
`;

            // 2. Fetch orbi/milp.py from the raw source
            // Note: In a real deployment, we'd bundle this.
            // For now, we fetch from GitHub raw or assuming strictly relative path in docs
            // Since we are in /docs, ../src/orbi/milp.py might not be served by GitHub Pages if not configured.
            // BUT: The user asked for "docs for this repository".
            // To be safe for GitHub Pages (which only serves /docs), I should perhaps COPY milp.py to docs/orbi_milp.py
            // OR I can inline it since I have access to it.
            // I'll assume we can fetch it, if not I'll handle error.
            // Actually, for this demo to work standalone in docs/, I should copy the necessary python files to docs/
            // or paste them into the virtual FS.
            // I will inline milp.py content here to be safe and robust!

            const orbiMilpCode = `
from ortools.linear_solver import pywraplp

class GRB:
    OPTIMAL = pywraplp.Solver.OPTIMAL
    INFEASIBLE = pywraplp.Solver.INFEASIBLE
    UNBOUNDED = pywraplp.Solver.UNBOUNDED
    FEASIBLE = pywraplp.Solver.FEASIBLE
    NOT_SOLVED = pywraplp.Solver.NOT_SOLVED
    CONTINUOUS = pywraplp.Solver.NumVar
    BINARY = pywraplp.Solver.BoolVar
    INTEGER = pywraplp.Solver.IntVar
    MINIMIZE = pywraplp.Solver.Minimize
    MAXIMIZE = pywraplp.Solver.Maximize
    INFINITY = pywraplp.Solver.infinity()

class OpenGurobiError(Exception):
    pass

class Var:
    def __init__(self, ortools_var, model_wrapper):
        self._var = ortools_var
        self._model = model_wrapper
        self.VarName = ortools_var.name()
        self.X = None
        self.RC = None

    def __str__(self):
        return f"<OpenGurobi Var {self.VarName}>"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return LinExpr(self) + other
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        return LinExpr(self) - other
    def __rsub__(self, other):
        return other + (self * -1)
    def __mul__(self, other):
        return LinExpr(self) * other
    def __rmul__(self, other):
        return self.__mul__(other)
    def __neg__(self):
        return self * -1

class LinExpr:
    def __init__(self, initial_expr=None, constant=0.0):
        self.terms = {}
        self.constant = constant
        if isinstance(initial_expr, Var):
            self.terms[initial_expr] = 1.0
        elif isinstance(initial_expr, (int, float)):
            self.constant = float(initial_expr)
        elif initial_expr is not None:
             raise OpenGurobiError(f"Cannot initialize LinExpr with type {type(initial_expr)}")

    def add(self, other, coeff=1.0):
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
        new_expr = LinExpr(other)
        new_expr.add(self, coeff=-1.0)
        return new_expr
    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise OpenGurobiError("Can only multiply LinExpr by a constant")
        new_expr = LinExpr()
        for var, coeff in self.terms.items():
            new_expr.terms[var] = coeff * other
        new_expr.constant = self.constant * other
        return new_expr
    def __rmul__(self, other):
        return self.__mul__(other)
    def __neg__(self):
        return self * -1
    def __le__(self, other):
        expr = self - other
        return (expr, '<=', 0)
    def __ge__(self, other):
        expr = self - other
        return (expr, '>=', 0)
    def equals(self, other):
        expr = self - other
        return (expr, '==', 0)
    def __eq__(self, other):
        return object.__eq__(self, other)

class Model:
    def __init__(self, name=""):
        self.solver = pywraplp.Solver.CreateSolver('CBC')
        if not self.solver:
            raise OpenGurobiError("Could not create CBC solver instance")
        self.name = name
        self._vars = {}
        self.Status = GRB.NOT_SOLVED
        self.ObjVal = None

    def addVar(self, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name=""):
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
        if obj != 0.0:
            self.solver.Objective().SetCoefficient(ort_var, obj)
        var_wrapper = Var(ort_var, self)
        self._vars[name] = var_wrapper
        return var_wrapper

    def setObjective(self, expr, sense=GRB.MINIMIZE):
        if isinstance(expr, Var):
            expr = LinExpr(expr)
        if not isinstance(expr, LinExpr):
             raise OpenGurobiError("Objective must be a LinExpr")
        objective = self.solver.Objective()
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
        if not (isinstance(constraint_tuple, tuple) and len(constraint_tuple) == 3):
             raise OpenGurobiError("Invalid constraint")
        expr, sense, rhs = constraint_tuple
        ort_expr = 0.0
        for var, coeff in expr.terms.items():
            ort_expr += var._var * coeff
        constant = expr.constant
        if sense == '<=':
            ub = rhs - constant
            ort_constr = self.solver.Add(ort_expr <= ub, name)
        elif sense == '>=':
            lb = rhs - constant
            ort_constr = self.solver.Add(ort_expr >= lb, name)
        elif sense == '==':
            val = rhs - constant
            ort_constr = self.solver.Add(ort_expr == val, name)
        else:
             raise OpenGurobiError(f"Unknown constraint sense {sense}")
        return ort_constr

    def optimize(self):
        print(f"Starting optimization with {self.solver.SolverVersion()}...")
        status = self.solver.Solve()
        self.Status = status
        if status == GRB.OPTIMAL:
            self.ObjVal = self.solver.Objective().Value()
            print(f"Optimal solution found. Objective: {self.ObjVal}")
            for name, var in self._vars.items():
                var.X = var._var.solution_value()
                var.RC = var._var.reduced_cost()
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
        if name not in self._vars:
             raise OpenGurobiError(f"Variable '{name}' not found.")
        return self._vars[name]

    def setParam(self, param_name, value):
        if param_name.lower() == "timelimit":
            self.solver.set_time_limit(int(value * 1000))

    def getVars(self):
        return self._vars.values()
            `;

            // Write files to virtual FS
            pyodide.FS.writeFile("mock_ortools.py", mockOrtoolsCode);

            // Create packages/modules
            await pyodide.runPythonAsync(`
import sys
import mock_ortools
sys.modules['ortools'] = mock_ortools
sys.modules['ortools.linear_solver'] = mock_ortools
sys.modules['ortools.linear_solver.pywraplp'] = mock_ortools
            `);

            // Create orbi package
            pyodide.FS.mkdir("orbi");
            pyodide.FS.writeFile("orbi/__init__.py", "");
            pyodide.FS.writeFile("orbi/milp.py", orbiMilpCode);

            loadingIndicator.innerText = "Python Environment Ready!";
            loadingIndicator.style.color = "#4ade80";
            pyodideReady = true;

        } catch (err) {
            loadingIndicator.innerText = "Error loading Python: " + err;
            loadingIndicator.style.color = "#ef4444";
            console.error(err);
        }
    }

    initPyodideEnv();

    // 1. Define Presets
    const PRESETS = {
        'basic': `from orbi.milp import Model, GRB

# Create a new model
m = Model("my_milp_model")

# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.INTEGER, lb=0, ub=10, name="z")

# Set objective: Maximize x + y + 2z
m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# Add constraints
m.addConstr(x + 2 * y + 3 * z <= 4, "c1")
m.addConstr(x + y >= 1, "c2")

# Optimize
m.optimize()

# Print results
if m.Status == GRB.OPTIMAL:
    print(f"Objective Value: {m.ObjVal}")
    for v in m.getVars():
        print(f"{v.VarName} = {v.X}")`,

        'infeasible': `from orbi.milp import Model, GRB

m = Model("infeasible")
x = m.addVar(name="x")
y = m.addVar(name="y")

m.addConstr(x + y <= 4)
m.addConstr(x + y >= 5)

m.optimize()

if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible")`,

        'unbounded': `from orbi.milp import Model, GRB

m = Model("unbounded")
x = m.addVar(name="x")

m.setObjective(x, GRB.MAXIMIZE)
# No upper bound on x

m.optimize()

if m.Status == GRB.UNBOUNDED:
    print("Model is unbounded")`,

        'knapsack': `from orbi.milp import Model, GRB

weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

m = Model("knapsack")
vars = []
for i in range(3):
    vars.append(m.addVar(vtype=GRB.BINARY, name=f"item_{i}"))

m.setObjective(sum(v*val for v, val in zip(vars, values)), GRB.MAXIMIZE)
m.addConstr(sum(v*w for v, w in zip(vars, weights)) <= capacity)

m.optimize()
print(f"Total Value: {m.ObjVal}")
for i, v in enumerate(vars):
    if v.X > 0.5:
        print(f"Item {i} selected")`
    };

    exampleSelect.addEventListener('change', (e) => {
        const key = e.target.value;
        if (PRESETS[key]) {
            codeBlock.innerText = PRESETS[key];
        }
    });

    // Initialize default
    if (codeBlock.innerText.trim().startsWith("from orbi.milp")) {
        // preserve
    } else {
        codeBlock.innerText = PRESETS['basic'];
    }

    runBtn.addEventListener('click', async () => {
        if (!pyodideReady) {
            alert("Python is still loading... please wait.");
            return;
        }

        const code = codeBlock.innerText;
        outputDiv.innerHTML = '<div id="loading-indicator" style="color:#4ade80">Python Environment Ready!</div><span class="prompt">$</span> python example.py<br>';

        runBtn.disabled = true;
        runBtn.style.opacity = '0.7';
        runBtn.innerText = 'Running...';

        try {
            // Rewrite print to capture output
            pyodide.runPython(`
import sys
import io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
            `);

            await pyodide.runPythonAsync(code);

            const stdout = pyodide.runPython("sys.stdout.getvalue()");
            const stderr = pyodide.runPython("sys.stderr.getvalue()");

            if (stdout) {
                const lines = stdout.split('\n');
                for (const line of lines) {
                    if (line) outputDiv.innerHTML += `<div>${line}</div>`;
                }
            }

            if (stderr) {
                const lines = stderr.split('\n');
                for (const line of lines) {
                    if (line) outputDiv.innerHTML += `<div style="color:#ef4444">${line}</div>`;
                }
            }

        } catch (err) {
            outputDiv.innerHTML += `<div style="color:#ef4444">${err}</div>`;
        } finally {
            runBtn.disabled = false;
            runBtn.style.opacity = '1';
            runBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Run Model
            `;
            outputDiv.innerHTML += '<br><span class="prompt">$</span> <span class="cursor">_</span>';
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }
    });
});
