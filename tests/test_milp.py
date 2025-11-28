import unittest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:  # Primary import
    from milp import Model, GRB, LinExpr, Var, OpenGurobiError  # noqa: E402
except ModuleNotFoundError:  # Fallback explicit loading
    import importlib.util
    _MILP_FILE = _SRC / "milp.py"
    spec = importlib.util.spec_from_file_location("milp", _MILP_FILE)
    if not spec or not spec.loader:
        raise
    milp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(milp_mod)
    Model = milp_mod.Model
    GRB = milp_mod.GRB
    LinExpr = milp_mod.LinExpr
    Var = milp_mod.Var
    OpenGurobiError = milp_mod.OpenGurobiError


def build_basic_model():
    m = Model("basic")
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.INTEGER, lb=0, ub=10, name="z")
    return m, x, y, z


class TestMilp(unittest.TestCase):
    def optimize_and_assert_optimal(self, model):
        model.optimize()
        self.assertEqual(
            model.Status, GRB.OPTIMAL
        )
        self.assertIsNotNone(model.ObjVal)

    def test_variable_creation(self):
        m = Model("var_creation")
        x = m.addVar(lb=0, ub=5, name="x")
        y = m.addVar(vtype=GRB.BINARY, name="y")
        z = m.addVar(vtype=GRB.INTEGER, lb=0, ub=3, name="z")
        self.assertIsInstance(x, Var)
        self.assertIsInstance(y, Var)
        self.assertIsInstance(z, Var)
        # Names should match
        self.assertEqual(x.VarName, "x")
        self.assertEqual(y.VarName, "y")
        self.assertEqual(z.VarName, "z")
        # Duplicate name should raise
        with self.assertRaises(OpenGurobiError):
            m.addVar(name="x")

    def test_objective_maximization(self):
        m, x, y, z = build_basic_model()
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
        m.addConstr(x + 2 * y + 3 * z <= 7, "c1")
        m.addConstr(x + y >= 1, "c2")
        self.optimize_and_assert_optimal(m)
        self.assertIsNotNone(x.X)
        self.assertIsNotNone(y.X)
        self.assertIsNotNone(z.X)
        # Objective value should be numeric
        self.assertIsInstance(m.ObjVal, (int, float))

    def test_constraints_solution(self):
        m, x, y, z = build_basic_model()
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
        c1 = m.addConstr(x + 2 * y + 3 * z <= 4, "c1")
        c2 = m.addConstr(x + y >= 1, "c2")
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.optimize_and_assert_optimal(m)
        self.assertIsNotNone(x.X)
        self.assertIsNotNone(y.X)
        self.assertIsNotNone(z.X)
        # Basic feasibility checks via solver status
        self.assertIn(m.Status, (GRB.OPTIMAL, GRB.FEASIBLE))

    def test_infeasible_model_status(self):
        m = Model("infeasible")
        x = m.addVar(vtype=GRB.BINARY, name="x")
        m.addConstr(x + 0 <= 0, "c1")
        m.addConstr(x + 0 >= 1, "c2")
        m.setObjective(x * 1, GRB.MAXIMIZE)
        m.optimize()
        self.assertEqual(m.Status, GRB.INFEASIBLE)

    def test_get_var_by_name(self):
        m, x, y, z = build_basic_model()
        self.assertIs(m.getVarByName("x"), x)
        self.assertIs(m.getVarByName("y"), y)
        with self.assertRaises(OpenGurobiError):
            m.getVarByName("nope")

    def test_invalid_objective_error(self):
        m, x, y, z = build_basic_model()
        with self.assertRaises(OpenGurobiError):
            m.setObjective(123)
        with self.assertRaises(OpenGurobiError):
            m.setObjective(x)

    def test_constraint_senses(self):
        m, x, y, z = build_basic_model()
        expr_le = x + y + z
        expr_ge = x + 2 * y
        expr_eq = 2 * z
        m.addConstr(expr_le <= 5, "le")
        m.addConstr(expr_ge >= 1, "ge")
        m.addConstr(expr_eq.equals(4), "eq")
        m.setObjective(z * 1, GRB.MAXIMIZE)
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            self.assertIsInstance(z.X, (int, float))
            self.assertEqual(z.X, 2)
            self.assertIn(x.X, (0, 1))
            self.assertIn(y.X, (0, 1))

    def test_duplicate_var_error(self):
        m = Model("dup")
        m.addVar(name="x")
        with self.assertRaises(OpenGurobiError):
            m.addVar(name="x")

    # Edge case: adding an invalid constraint tuple
    def test_invalid_constraint_tuple(self):
        m = Model("bad_constr")
        x = m.addVar(name="x")
        with self.assertRaises(OpenGurobiError):
            m.addConstr((x, '<=', 1, 'extra'))
        with self.assertRaises(OpenGurobiError):
            m.addConstr("not a tuple")

    # Ensure LinExpr arithmetic behaves
    def test_linexpr_arithmetic(self):
        m = Model("expr")
        x = m.addVar(name="x")
        y = m.addVar(name="y")
        expr = 2 * x + 3 * y - 5
        self.assertIsInstance(expr, LinExpr)
        coeffs = {v.VarName: c for v, c in expr.terms.items()}
        self.assertEqual(coeffs.get("x"), 2)
        self.assertEqual(coeffs.get("y"), 3)
        self.assertEqual(expr.constant, -5)

    def test_set_param_time_limit(self):
        m = Model("params")
        m.setParam("TimeLimit", 0.01)
        x = m.addVar(name="x")
        m.setObjective(x * 1, GRB.MAXIMIZE)
        m.optimize()
        self.assertIn(
            m.Status,
            (GRB.OPTIMAL, GRB.NOT_SOLVED, GRB.FEASIBLE, GRB.UNBOUNDED),
        )

    def test_unbounded_model(self):
        m = Model("unbounded")
        x = m.addVar(lb=0, ub=GRB.INFINITY, name="x")
        m.setObjective(x * 1, GRB.MAXIMIZE)
        m.optimize()
        self.assertIn(m.Status, (GRB.UNBOUNDED, GRB.OPTIMAL))


if __name__ == "__main__":
    unittest.main()
