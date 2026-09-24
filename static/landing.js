function displayMetric(value, digits = 3) {
    return value === null || value === undefined || Number.isNaN(Number(value))
        ? "Not recorded"
        : Number(value).toFixed(digits);
}

async function loadLatestEvidence() {
    const preview = document.querySelector(".evidence-preview");
    try {
        const response = await fetch("/api/runs", {cache: "no-store"});
        if (!response.ok) throw new Error("Results could not be loaded");
        const runs = await response.json();
        const run = runs.find(item => item.status === "completed");

        preview.classList.remove("is-loading");
        preview.setAttribute("aria-busy", "false");
        if (!run) {
            preview.classList.add("is-empty");
            document.getElementById("preview-run-id").textContent = "No runs";
            document.getElementById("preview-name").textContent = "No completed evidence yet";
            document.getElementById("preview-context").textContent = "Your first completed experiment will appear here automatically.";
            document.getElementById("preview-mrr").textContent = "Not recorded";
            document.getElementById("preview-hits").textContent = "Not recorded";
            document.getElementById("preview-ece").textContent = "Not recorded";
            document.getElementById("preview-protocol").textContent = "Ready for a benchmark";
            document.getElementById("preview-seed").textContent = "Fixed seed recorded per run";
            return;
        }

        document.getElementById("preview-run-id").textContent = `Run #${run.id}`;
        document.getElementById("preview-name").textContent = run.experiment_name;
        document.getElementById("preview-context").textContent = `${run.model_type} on ${run.dataset}`;
        document.getElementById("preview-mrr").textContent = displayMetric(run.mrr);
        document.getElementById("preview-hits").textContent = displayMetric(run.hits_at_10);
        document.getElementById("preview-ece").textContent = displayMetric(run.ece);
        document.getElementById("preview-protocol").textContent = run.full_evaluation ? "Full evaluation" : `Sample of ${run.eval_sample_size}`;
        document.getElementById("preview-seed").textContent = `Seed ${run.seed}`;
    } catch (_) {
        preview.classList.remove("is-loading");
        preview.classList.add("is-empty");
        preview.setAttribute("aria-busy", "false");
        document.getElementById("preview-run-id").textContent = "Offline";
        document.getElementById("preview-name").textContent = "Tracked evidence is unavailable";
        document.getElementById("preview-context").textContent = "Start the local service, then reload this page.";
        document.getElementById("preview-mrr").textContent = "Not recorded";
        document.getElementById("preview-hits").textContent = "Not recorded";
        document.getElementById("preview-ece").textContent = "Not recorded";
        document.getElementById("preview-protocol").textContent = "Local service unavailable";
        document.getElementById("preview-seed").textContent = "No data loaded";
    }
}

document.querySelector(".evidence-preview").classList.add("is-loading");
loadLatestEvidence();
