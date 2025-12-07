
document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const outputDiv = document.getElementById('output');

    const expectedOutput = [
        "--- Testing Open-Source Gurobi-like API ---",
        "Loading model...",
        "Solver: CBC",
        "Variables: 3",
        "Constraints: 2",
        "Optimizing...",
        "...",
        "",
        "--- Solution ---",
        "Objective Value: 31.0",
        "x = 1.0",
        "y = 0.0",
        "z = 10.0",
        "",
        "Process finished with exit code 0"
    ];

    runBtn.addEventListener('click', async () => {
        // Clear previous run (keep the prompt)
        outputDiv.innerHTML = '<span class="prompt">$</span> python example.py<br>';

        // Disable button
        runBtn.disabled = true;
        runBtn.style.opacity = '0.7';
        runBtn.innerText = 'Running...';

        for (const line of expectedOutput) {
            await new Promise(r => setTimeout(r, 200 + Math.random() * 300));
            outputDiv.innerHTML += `<div>${line}</div>`;
            // auto scroll
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }

        // Reset button
        runBtn.disabled = false;
        runBtn.style.opacity = '1';
        runBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            Run Model
        `;

        // Add new prompt
        outputDiv.innerHTML += '<br><span class="prompt">$</span> <span class="cursor">_</span>';
        outputDiv.scrollTop = outputDiv.scrollHeight;
    });
});
