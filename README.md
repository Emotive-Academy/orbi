# LibreMIP 

`LibreMIP` (Open-Source Gurobi-like API)

A proof-of-concept Python package that provides a Gurobi-like modeling API (gurobipy) using Google OR-Tools as the backend solver.

This project is intended for users who are familiar with the expressive and convenient modeling syntax of gurobipy but want to use a powerful, free, and open-source solver like Google OR-Tools (which uses the CBC solver for MILPs).

## Why?

The gurobipy API is excellent for building optimization models, allowing for natural, algebra-like syntax:

```
m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
m.addConstr(x + 2 * y + 3 * z <= 4, "c1")
```

This wrapper re-implements that gurobipy-style API by creating wrapper classes (Model, Var, LinExpr) that translate these simple expressions into the corresponding objects and calls for the Google OR-Tools backend.

## Features

This is a lightweight wrapper and not a complete replacement, but it supports the most common modeling objects and methods:

- Constants: `GRB.OPTIMAL`, `GRB.BINARY`, `GRB.INTEGER`, `GRB.MAXIMIZE`, etc.
- Model: `Model()`
- Variables: `model.addVar()`
- Linear Expressions: `LinExpr` class with support for +, -, *.
- Constraints: `model.addConstr()` using overloaded operators (<=, >=, ==).
- Objective: `model.setObjective()`
- Solving: `model.optimize()`
- Parameters: A basic `model.setParam()` (e.g., for TimeLimit).
- Results: Access solution values via `model.ObjVal` and `var.X`.

## Installation

You only need Google OR-Tools, as this package is just a single-file wrapper.

## Example Usage

The following code (from opensource_gurobi_api.py) demonstrates how to use the wrapper. Notice how the code looks almost identical to gurobipy.

```python
# Import the wrapper classes
from milp import Model, GRB

print("--- Testing Open-Source Gurobi-like API ---")
# Create a new model

m = Model("my_milp_model")

# --- Create variables ---
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

# --- Set parameters ---
m.setParam("TimeLimit", 60) # 60 seconds

# --- Optimize the model ---
m.optimize()

# --- Print results ---
if m.Status == GRB.OPTIMAL:
    print("\n--- Solution ---")
    print(f"Objective Value: {m.ObjVal}")
    for v in m._vars.values():
        print(f"{v.VarName} = {v.X}")
else:
    print("No optimal solution found.")
```

# Disclaimer

This is a demonstration and not a complete, production-ready drop-in replacement for gurobipy. Many features, attributes, and parameters are not implemented. It is meant to serve as a starting point and a proof-of-concept.
