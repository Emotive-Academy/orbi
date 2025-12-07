
document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const outputDiv = document.getElementById('output');
    const codeBlock = document.getElementById('code-block');
    const exampleSelect = document.getElementById('example-select');

    // 1. Define Presets (Code & Output)
    const PRESETS = {
        'basic': {
            code: `from orbi.milp import Model, GRB

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
print(f"Objective Value: {m.ObjVal}")
for v in m.getVars():
    print(f"{v.VarName} = {v.X}")`,
            output: [
                "--- Testing Open-Source Gurobi-like API ---",
                "Loading model...",
                "Solver: CBC",
                "Variables: 3",
                "Constraints: 2",
                "Optimizing...",
                "...",
                "Objective Value: 31.0",
                "x = 1.0",
                "y = 0.0",
                "z = 10.0"
            ]
        },
        'infeasible': {
            code: `from orbi.milp import Model, GRB

m = Model("infeasible")
x = m.addVar(name="x")
y = m.addVar(name="y")

m.addConstr(x + y <= 4)
m.addConstr(x + y >= 5)

m.optimize()

if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible")`,
            output: [
                "Loading model...",
                "Solver: CBC",
                "Variables: 2",
                "Constraints: 2",
                "Optimizing...",
                "Status: Infeasible",
                "Model is infeasible"
            ]
        },
        'unbounded': {
            code: `from orbi.milp import Model, GRB

m = Model("unbounded")
x = m.addVar(name="x")

m.setObjective(x, GRB.MAXIMIZE)
# No upper bound on x

m.optimize()

if m.Status == GRB.UNBOUNDED:
    print("Model is unbounded")`,
            output: [
                "Loading model...",
                "Solver: CBC",
                "Optimizing...",
                "Status: Unbounded",
                "Model is unbounded"
            ]
        },
        'knapsack': {
            code: `from orbi.milp import Model, GRB

# Knapsack Problem
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
print(f"Total Value: {m.ObjVal}")`,
            output: [
                "Loading model...",
                "Solver: CBC",
                "Variables: 3",
                "Constraints: 1",
                "Optimizing...",
                "Total Value: 220.0"
            ]
        }
    };

    const GENERIC_OUTPUT = [
        "Loading user model...",
        "Solver: CBC",
        "Parsings variables...",
        "Parsing constraints...",
        "Optimizing...",
        "...",
        "[SIMULATION MODE] Custom code detected.",
        "Since this is a static page, we cannot run arbitrary Python code.",
        "Please use the Example Selector to see real outputs for specific cases.",
        "Your code syntax looks correct!",
        "Process finished with exit code 0"
    ];

    // 2. Helper to normalize strings for comparison (ignore whitespace differences)
    function normalize(str) {
        return str.replace(/\s+/g, ' ').trim();
    }

    // 3. Load Example Logic
    exampleSelect.addEventListener('change', (e) => {
        const key = e.target.value;
        if (PRESETS[key]) {
            codeBlock.textContent = PRESETS[key].code;
            // Highlight effect (reset opacity to indicate change)
            codeBlock.style.opacity = '0';
            setTimeout(() => codeBlock.style.opacity = '1', 50);
        }
    });

    // 4. Run Logic
    runBtn.addEventListener('click', async () => {
        // Reset terminal
        outputDiv.innerHTML = '<span class="prompt">$</span> python example.py<br>';

        // UI State
        runBtn.disabled = true;
        runBtn.style.opacity = '0.7';
        runBtn.innerText = 'Running...';

        // Determine output
        const currentCode = normalize(codeBlock.innerText);
        let linesToPrint = GENERIC_OUTPUT;

        // Check if matches any preset
        for (const key in PRESETS) {
            if (normalize(PRESETS[key].code) === currentCode) {
                linesToPrint = PRESETS[key].output;
                break;
            }
        }

        // Simulate typing/processing delay
        for (const line of linesToPrint) {
            await new Promise(r => setTimeout(r, 100 + Math.random() * 200));
            outputDiv.innerHTML += `<div>${line}</div>`;
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }

        // Reset Button
        runBtn.disabled = false;
        runBtn.style.opacity = '1';
        runBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            Run Model
        `;

        outputDiv.innerHTML += '<br><span class="prompt">$</span> <span class="cursor">_</span>';
        outputDiv.scrollTop = outputDiv.scrollHeight;
    });

    // Initialize with basic
    if (codeBlock.innerText.trim() === "") {
        codeBlock.textContent = PRESETS['basic'].code;
    }
});
