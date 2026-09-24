const comparisonMetrics = [
    ["mean_rank", "Mean Rank", "lower"],
    ["mrr", "MRR", "higher"],
    ["hits_at_1", "Hits@1", "higher"],
    ["hits_at_3", "Hits@3", "higher"],
    ["hits_at_10", "Hits@10", "higher"],
    ["precision", "Precision", "higher"],
    ["recall", "Recall", "higher"],
    ["f1_score", "F1", "higher"],
    ["mse", "MSE", "lower"],
    ["mae", "MAE", "lower"],
    ["ece", "ECE", "lower"],
    ["brier", "Brier", "lower"],
    ["nll", "NLL", "lower"]
];

const runLabels = ["A", "B", "C", "D"];

function comparisonMetricsForRuns(runs) {
    const definitions = new Map(comparisonMetrics.map(item => [item[0], item]));
    runs.forEach(run => Object.entries(run.metric_definitions || {}).forEach(([key, definition]) => {
        definitions.set(key, [key, definition.label || key.replaceAll("_", " "), definition.direction || "higher"]);
    }));
    return [...definitions.values()].filter(([key]) => runs.some(run => run[key] !== null && run[key] !== undefined));
}

const protocolFields = [
    ["Dataset", run => run.dataset],
    ["Dataset plugin version", run => run.dataset_plugin_version || "legacy"],
    ["Model plugin version", run => run.model_plugin_version || "legacy"],
    ["Evaluation protocol", run => run.evaluation_protocol_version || "legacy"],
    ["Evaluators", run => (run.evaluators || []).map(item => `${item.id}@${item.version}`).join(", ") || "legacy"],
    ["Dataset fingerprint", run => run.dataset_fingerprint || "unrecorded"],
    ["Seed", run => run.seed],
    ["Evaluation scope", run => run.full_evaluation ? "Full" : `Sample ${run.eval_sample_size}`],
    ["Training sample", run => run.train_sample_size || "Full"],
    ["Epochs", run => run.config?.epochs],
    ["Batch size", run => run.config?.batch_size],
    ["Embedding dimension", run => run.config?.embedding_dimension],
    ["Distance norm", run => `L${run.config?.p_norm}`],
    ["Uncertainty level", run => run.config?.uncertainty_level]
];

function formatMetric(value, digits = 4) {
    return value === null || value === undefined || Number.isNaN(Number(value))
        ? "—"
        : Number(value).toFixed(digits);
}

function formatCount(value) {
    return value === null || value === undefined ? "—" : Number(value).toLocaleString();
}

function numericValue(value) {
    return value === null || value === undefined || value === "" ? Number.NaN : Number(value);
}

function element(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== "") node.textContent = text;
    return node;
}

function showError(message) {
    const loading = document.getElementById("comparison-loading");
    loading.classList.add("hidden");
    loading.setAttribute("aria-busy", "false");
    document.getElementById("comparison-content").classList.add("hidden");
    document.getElementById("comparison-error-message").textContent = message;
    document.getElementById("comparison-error").classList.remove("hidden");
}

function renderRunCards(runs) {
    const pair = document.getElementById("run-pair");
    pair.textContent = "";
    runs.forEach((run, index) => {
        const runClass = runLabels[index].toLowerCase();
        const card = element("article", `run-summary run-${runClass}`);
        const top = element("div", "run-summary-top");
        const identity = element("div");
        identity.append(element("span", "run-label", `RUN ${runLabels[index]} · #${run.id}`));
        identity.append(element("h2", "", run.experiment_name));
        identity.append(element("p", "", `${run.model_type} · ${run.dataset} · seed ${run.seed}`));
        const model = element("span", "model-badge", run.model_type);
        top.append(identity, model);

        const metrics = element("div", "run-summary-metrics");
        [["MRR", run.mrr], ["F1", run.f1_score], ["ECE", run.ece]].forEach(([label, value]) => {
            const item = element("div");
            item.append(element("small", "", label), element("strong", "", formatMetric(value, 3)));
            metrics.appendChild(item);
        });
        card.append(top, metrics);
        pair.appendChild(card);
    });
}

function renderProtocol(runs) {
    const differences = protocolFields.filter(([, getter]) => new Set(runs.map(run => String(getter(run)))).size > 1);
    const box = document.getElementById("protocol-status");
    box.textContent = "";
    const icon = element("span", "protocol-status-icon", differences.length ? "!" : "✓");
    icon.setAttribute("aria-hidden", "true");
    const copy = element("div");
    copy.append(
        element("strong", "", differences.length ? "Protocol differences detected" : "Matched comparison protocol"),
        element("p", "", differences.length
            ? "Interpret model differences carefully because the selected runs were not executed under identical conditions."
            : "The runs share the same dataset, seed, budget, evaluation scope, and uncertainty setting.")
    );
    box.className = `protocol-status ${differences.length ? "warning" : "matched"}`;
    box.append(icon, copy);
    if (differences.length) {
        const list = element("div", "protocol-differences");
        differences.forEach(([label, getter]) => {
            const chip = element("span");
            const values = runs.map((run, index) => `${runLabels[index]}: ${getter(run)}`).join(" · ");
            chip.append(element("b", "", label), document.createTextNode(` ${values}`));
            list.appendChild(chip);
        });
        box.appendChild(list);
    }
}

function bestRunIndexes(runs, key, direction) {
    const available = runs
        .map((run, index) => ({index, value: numericValue(run[key])}))
        .filter(item => Number.isFinite(item.value));
    if (!available.length) return [];
    const bestValue = direction === "higher"
        ? Math.max(...available.map(item => item.value))
        : Math.min(...available.map(item => item.value));
    return available.filter(item => item.value === bestValue).map(item => item.index);
}

function directionLabel(runs, key, direction, compact = false) {
    const validCount = runs.filter(run => Number.isFinite(numericValue(run[key]))).length;
    const winners = bestRunIndexes(runs, key, direction);
    const text = compact
        ? (direction === "higher" ? "↑ Higher" : "↓ Lower")
        : `${direction} is better`;
    const label = element(compact ? "td" : "small", `direction direction-${direction}`, text);
    if (validCount >= 2 && winners.length === 1) {
        const winner = runLabels[winners[0]];
        label.classList.add(`direction-winner-${winner.toLowerCase()}`);
        label.setAttribute("aria-label", `${text}. Run ${winner} performs best.`);
        label.title = `Run ${winner} performs best`;
    } else if (validCount >= 2 && winners.length > 1) {
        const tied = winners.map(index => `Run ${runLabels[index]}`).join(" and ");
        label.setAttribute("aria-label", `${text}. ${tied} are tied.`);
        label.title = `${tied} are tied`;
    } else {
        label.setAttribute("aria-label", `${text}. A winner cannot be determined.`);
        label.title = "A winner cannot be determined";
    }
    return label;
}

function renderInsights(runs) {
    const grid = document.getElementById("insight-grid");
    grid.textContent = "";
    const definitions = [
        ["Ranking leader", "mrr", "higher", "MRR", "The stronger reciprocal-rank result."],
        ["Calibration leader", "ece", "lower", "ECE", "Confidence is closer to observed outcomes."],
        ["Faster execution", "total_seconds", "lower", "seconds", "Lower end-to-end benchmark time."]
    ];
    definitions.forEach(([title, key, direction, unit, note]) => {
        const winners = bestRunIndexes(runs, key, direction);
        const card = element("article", "insight-card");
        card.append(element("small", "", title));
        if (!winners.length) {
            card.append(element("strong", "", "Not available"), element("p", "", note));
        } else if (winners.length > 1) {
            const labels = winners.map(index => runLabels[index]).join(", ");
            card.append(element("strong", "", `Tie · Runs ${labels}`), element("p", "", `${formatMetric(runs[winners[0]][key], key === "total_seconds" ? 2 : 4)} ${unit}`));
        } else {
            const winner = winners[0];
            card.append(
                element("strong", `insight-winner winner-${runLabels[winner].toLowerCase()}`, `Run ${runLabels[winner]}`),
                element("p", "", `${formatMetric(runs[winner][key], key === "total_seconds" ? 2 : 4)} ${unit} · ${note}`)
            );
        }
        grid.appendChild(card);
    });
}

function renderMetricChart(containerId, runs, metrics, digits = 4) {
    const container = document.getElementById(containerId);
    container.textContent = "";
    metrics.filter(([key]) => runs.some(run => Number.isFinite(numericValue(run[key])))).forEach(([key, label, direction]) => {
        const values = runs.map(run => numericValue(run[key]));
        const available = values.filter(value => Number.isFinite(value));
        const scale = Math.max(...available, 0.000001);
        const row = element("div", "metric-chart-row");
        const heading = element("div", "metric-chart-label");
        heading.append(element("strong", "", label), directionLabel(runs, key, direction));
        row.appendChild(heading);
        const seriesGrid = element("div", "metric-series-grid");
        values.forEach((value, index) => {
            const runClass = runLabels[index].toLowerCase();
            const series = element("div", `metric-series series-${runClass}`);
            const meta = element("div", "metric-series-meta");
            meta.append(element("span", "", `Run ${runLabels[index]}`), element("b", "", Number.isFinite(value) ? formatMetric(value, digits) : "—"));
            const track = element("div", "metric-track");
            const bar = element("span", "metric-bar");
            bar.style.width = Number.isFinite(value) ? `${Math.max(2, (value / scale) * 100)}%` : "0%";
            track.appendChild(bar);
            series.append(meta, track);
            seriesGrid.appendChild(series);
        });
        row.appendChild(seriesGrid);
        container.appendChild(row);
    });
}

function datasetCard(info, run, label, wide = false) {
    const card = element("article", `dataset-card${wide ? " shared" : ""}`);
    const heading = element("div", "dataset-card-heading");
    const title = element("div");
    title.append(element("span", "dataset-run-label", label), element("h3", "", info.name));
    heading.append(title, element("span", "dataset-source", info.source || "Run metadata"));
    card.append(heading, element("p", "dataset-description", info.description || "Knowledge graph benchmark dataset."));

    const stats = element("div", "dataset-stats");
    [
        ["Entities", info.entities], ["Relations", info.relations],
        ["Total triples", info.total_triples], ["Train", info.train_triples],
        ["Validation", info.validation_triples], ["Test", info.test_triples]
    ].forEach(([name, value]) => {
        const item = element("div");
        item.append(element("small", "", name), element("strong", "", formatCount(value)));
        stats.appendChild(item);
    });
    card.appendChild(stats);

    const splitValues = [info.train_triples, info.validation_triples, info.test_triples].map(value => Number(value) || 0);
    const total = splitValues.reduce((sum, value) => sum + value, 0);
    if (total) {
        const split = element("div", "dataset-split");
        const labels = element("div", "dataset-split-labels");
        labels.append(element("span", "", "Train"), element("span", "", "Validation"), element("span", "", "Test"));
        const bar = element("div", "dataset-split-bar");
        ["train", "validation", "test"].forEach((name, index) => {
            const segment = element("span", name);
            segment.style.width = `${(splitValues[index] / total) * 100}%`;
            bar.appendChild(segment);
        });
        split.append(labels, bar);
        card.appendChild(split);
    }
    const usage = wide
        ? "All selected runs use this dataset; evaluation scope is compared in the protocol section above."
        : run.full_evaluation
            ? "This run evaluated the complete validation and test splits."
            : `This run evaluated a sample of ${formatCount(run.eval_sample_size)} triples per evaluation split.`;
    card.appendChild(element("p", "dataset-usage", usage));
    return card;
}

function renderDatasets(infos, runs) {
    const grid = document.getElementById("dataset-grid");
    grid.textContent = "";
    if (new Set(runs.map(run => run.dataset)).size === 1) {
        grid.appendChild(datasetCard(infos[0], runs[0], "SHARED DATASET", true));
    } else {
        runs.forEach((run, index) => grid.appendChild(datasetCard(infos[index], run, `RUN ${runLabels[index]} DATASET`)));
    }
}

function renderMetricTable(runs) {
    const head = document.getElementById("metric-table-head");
    const body = document.getElementById("metric-table-body");
    head.textContent = "";
    body.textContent = "";
    const headRow = element("tr");
    const headings = ["Metric", ...runs.map((run, index) => `Run ${runLabels[index]} · ${run.model_type}`), "Direction"];
    if (runs.length === 2) headings.push("Difference (B - A)");
    headings.forEach(label => headRow.appendChild(element("th", "", label)));
    head.appendChild(headRow);
    comparisonMetricsForRuns(runs).forEach(([key, label, direction]) => {
        const row = element("tr");
        const winners = bestRunIndexes(runs, key, direction);
        row.appendChild(element("td", "metric-name", label));
        runs.forEach((run, index) => {
            const classes = winners.includes(index) ? `table-winner run-${runLabels[index].toLowerCase()}-color` : "";
            row.appendChild(element("td", classes, formatMetric(run[key])));
        });
        row.appendChild(directionLabel(runs, key, direction, true));
        if (runs.length === 2) {
            const a = numericValue(runs[0][key]);
            const b = numericValue(runs[1][key]);
            row.appendChild(element("td", "mono", Number.isFinite(a) && Number.isFinite(b) ? `${b - a >= 0 ? "+" : ""}${formatMetric(b - a)}` : "—"));
        }
        body.appendChild(row);
    });
}

async function loadComparison() {
    const rawIds = new URLSearchParams(window.location.search).get("ids") || "";
    const ids = [...new Set(rawIds.split(",").map(value => Number(value.trim())).filter(Number.isInteger))];
    if (ids.length < 2 || ids.length > 4) {
        showError("Choose between two and four completed runs from the tracking page.");
        return;
    }
    try {
        const responses = await Promise.all(ids.flatMap(id => [fetch(`/api/runs/${id}`), fetch(`/api/runs/${id}/dataset-info`)]));
        if (responses.some(response => !response.ok)) throw new Error("One or more selected runs could not be loaded.");
        const payloads = await Promise.all(responses.map(response => response.json()));
        const runs = payloads.filter((_, index) => index % 2 === 0);
        const infos = payloads.filter((_, index) => index % 2 === 1);
        if (runs.some(run => run.status !== "completed")) throw new Error("Only completed runs can be compared.");

        document.getElementById("comparison-id").textContent = runs.map(run => `#${run.id}`).join(" ↔ ");
        renderRunCards(runs);
        renderProtocol(runs);
        renderInsights(runs);
        renderMetricChart("ranking-chart", runs, [["mrr", "MRR", "higher"], ["hits_at_1", "Hits@1", "higher"], ["hits_at_3", "Hits@3", "higher"], ["hits_at_10", "Hits@10", "higher"], ["f1_score", "F1", "higher"]]);
        renderMetricChart("calibration-chart", runs, [["ece", "ECE", "lower"], ["brier", "Brier", "lower"], ["nll", "NLL", "lower"], ["mae", "MAE", "lower"]]);
        renderMetricChart("timing-chart", runs, [["train_seconds", "Training", "lower"], ["eval_seconds", "Evaluation", "lower"], ["total_seconds", "Total", "lower"]], 2);
        renderDatasets(infos, runs);
        renderMetricTable(runs);
        const loading = document.getElementById("comparison-loading");
        loading.classList.add("hidden");
        loading.setAttribute("aria-busy", "false");
        document.getElementById("comparison-content").classList.remove("hidden");
    } catch (error) {
        showError(error.message || "Could not build this comparison.");
    }
}

loadComparison();
