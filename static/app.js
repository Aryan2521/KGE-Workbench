const sectionCopy = {
    "run-section": ["Run a real benchmark", "Execute an installed model plugin with a recorded configuration and fixed seed."],
    "results-section": ["Run details", "Inspect ranking, calibration, classification, timing, and provenance."],
    "tracking-section": ["Tracking and comparison", "Filter, compare, and export reproducible experiment runs."]
};

const metricDefinitions = [
    ["mean_rank", "Mean Rank", "lower"], ["mrr", "MRR", "higher"],
    ["hits_at_1", "Hits@1", "higher"], ["hits_at_3", "Hits@3", "higher"],
    ["hits_at_10", "Hits@10", "higher"], ["precision", "Precision", "higher"],
    ["recall", "Recall", "higher"], ["f1_score", "F1", "higher"],
    ["mse", "MSE", "lower"], ["mae", "MAE", "lower"],
    ["ece", "ECE", "lower"], ["brier", "Brier", "lower"], ["nll", "NLL", "lower"]
];

function metricDefinitionsForRun(run) {
    const definitions = run.metric_definitions || {};
    if (Object.keys(definitions).length) {
        return Object.entries(definitions).map(([key, definition]) => [
            key,
            definition.label || key.replaceAll("_", " "),
            definition.direction || "higher"
        ]);
    }
    return metricDefinitions.filter(([key]) => run[key] !== null && run[key] !== undefined);
}

let runCache = [];
let activeEvents = null;
let pendingDeleteMode = null;
let pluginCatalog = {models: [], datasets: [], evaluators: [], core_parameters: {}};
let trackingRefreshTimer = null;
let runsRequestInFlight = false;

function stopTrackingRefresh() {
    if (trackingRefreshTimer) clearInterval(trackingRefreshTimer);
    trackingRefreshTimer = null;
}

function updateTrackingRefresh() {
    const trackingVisible = document.getElementById("tracking-section").classList.contains("active");
    const hasActiveRuns = runCache.some(run => run.status === "queued" || run.status === "running");
    if (trackingVisible && hasActiveRuns && !trackingRefreshTimer) {
        trackingRefreshTimer = setInterval(() => loadRuns(), 1500);
    } else if ((!trackingVisible || !hasActiveRuns) && trackingRefreshTimer) {
        stopTrackingRefresh();
    }
}

function switchSection(sectionId) {
    document.querySelectorAll(".tab-content").forEach(element => element.classList.toggle("active", element.id === sectionId));
    document.querySelectorAll(".nav-item").forEach(element => {
        const active = element.dataset.section === sectionId;
        element.classList.toggle("active", active);
        element.setAttribute("aria-selected", String(active));
        element.tabIndex = active ? 0 : -1;
    });
    document.getElementById("current-section-title").textContent = sectionCopy[sectionId][0];
    document.getElementById("current-section-desc").textContent = sectionCopy[sectionId][1];
    if (sectionId === "tracking-section") loadRuns();
    else stopTrackingRefresh();
}

function numberValue(id) { return Number(document.getElementById(id).value); }
function formatMetric(value, digits = 4) { return value === null || value === undefined ? "—" : Number(value).toFixed(digits); }
function shortHash(value) { return value ? `${value.slice(0, 12)}…` : "—"; }

function updateConditionalFields() {
    document.getElementById("custom-uncertainty").classList.toggle("hidden", document.getElementById("uncertainty_level").value !== "custom");
}

function pluginById(kind, id) { return pluginCatalog[kind].find(plugin => plugin.id === id); }

function renderPluginParameters(kind) {
    const select = document.getElementById(kind === "models" ? "model_type" : "dataset");
    const plugin = pluginById(kind, select.value);
    const fields = document.getElementById(kind === "models" ? "model-parameter-fields" : "dataset-parameter-fields");
    const section = document.getElementById(kind === "models" ? "model-plugin-settings" : "dataset-plugin-settings");
    const description = document.getElementById(kind === "models" ? "model-plugin-description" : "dataset-plugin-description");
    fields.textContent = "";
    if (!plugin) return;
    description.textContent = plugin.description || "Installed local plugin.";
    Object.entries(plugin.parameters || {}).forEach(([name, schema]) => {
        const group = document.createElement("div");
        group.className = "form-group col-half";
        const label = document.createElement("label");
        const inputId = `plugin-${kind}-${name}`;
        label.htmlFor = inputId;
        label.textContent = schema.label || name.replaceAll("_", " ");
        let input;
        if (schema.enum) {
            input = document.createElement("select");
            schema.enum.forEach(value => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = String(value).replaceAll("_", " ");
                input.appendChild(option);
            });
        } else {
            input = document.createElement("input");
            input.type = schema.type === "integer" || schema.type === "number" ? "number" : schema.type === "boolean" ? "checkbox" : "text";
            if (schema.minimum !== undefined) input.min = schema.minimum;
            if (schema.maximum !== undefined) input.max = schema.maximum;
            input.step = schema.step ?? (schema.type === "number" ? "any" : "1");
        }
        input.id = inputId;
        input.dataset.pluginParam = name;
        input.dataset.pluginType = schema.type || "string";
        input.dataset.pluginKind = kind;
        if (schema.type === "boolean") input.checked = Boolean(schema.default);
        else if (schema.default !== undefined) input.value = schema.default;
        group.append(label, input);
        fields.appendChild(group);
    });
    section.classList.toggle("hidden", !Object.keys(plugin.parameters || {}).length);
}

function collectPluginParameters(config) {
    const evaluatorParameters = {};
    document.querySelectorAll("[data-plugin-param]").forEach(input => {
        let value;
        if (input.dataset.pluginType === "boolean") value = input.checked;
        else if (["integer", "number"].includes(input.dataset.pluginType)) value = Number(input.value);
        else value = input.value;
        if (input.dataset.pluginKind === "evaluators") {
            evaluatorParameters[input.dataset.pluginId] ||= {};
            evaluatorParameters[input.dataset.pluginId][input.dataset.pluginParam] = value;
        } else {
            config[input.dataset.pluginParam] = value;
        }
    });
    config.evaluator_parameters = evaluatorParameters;
    return config;
}

function evaluatorCompatibility(evaluator, model, dataset) {
    const capabilities = model?.capabilities || {};
    const features = new Set(dataset?.features || []);
    const missingCapabilities = (evaluator.required_model_capabilities || []).filter(name => !capabilities[name]);
    const missingFeatures = (evaluator.required_dataset_features || []).filter(name => !features.has(name));
    return {compatible: !missingCapabilities.length && !missingFeatures.length, missingCapabilities, missingFeatures};
}

function createParameterInput(kind, plugin, name, schema) {
    const group = document.createElement("div");
    group.className = "form-group col-half";
    const label = document.createElement("label");
    const inputId = `plugin-${kind}-${plugin.id}-${name}`;
    label.htmlFor = inputId;
    label.textContent = schema.label || name.replaceAll("_", " ");
    let input;
    if (schema.enum) {
        input = document.createElement("select");
        schema.enum.forEach(value => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = String(value).replaceAll("_", " ");
            input.appendChild(option);
        });
    } else {
        input = document.createElement("input");
        input.type = schema.type === "integer" || schema.type === "number" ? "number" : schema.type === "boolean" ? "checkbox" : "text";
        if (schema.minimum !== undefined) input.min = schema.minimum;
        if (schema.maximum !== undefined) input.max = schema.maximum;
        input.step = schema.step ?? (schema.type === "number" ? "any" : "1");
    }
    input.id = inputId;
    input.dataset.pluginParam = name;
    input.dataset.pluginType = schema.type || "string";
    input.dataset.pluginKind = kind;
    input.dataset.pluginId = plugin.id;
    if (schema.type === "boolean") input.checked = Boolean(schema.default);
    else if (schema.default !== undefined) input.value = schema.default;
    group.append(label, input);
    return group;
}

function renderEvaluatorParameters() {
    const fields = document.getElementById("evaluator-parameter-fields");
    const selected = [...document.querySelectorAll("[data-evaluator-id]:checked")].map(input => input.dataset.evaluatorId);
    fields.textContent = "";
    selected.forEach(id => {
        const plugin = pluginById("evaluators", id);
        if (!plugin || !Object.keys(plugin.parameters || {}).length) return;
        const group = document.createElement("div");
        group.className = "evaluator-parameter-group";
        Object.entries(plugin.parameters).forEach(([name, schema]) => group.appendChild(createParameterInput("evaluators", plugin, name, schema)));
        fields.appendChild(group);
    });
    document.getElementById("evaluator-plugin-settings").classList.toggle("hidden", !fields.children.length);
    document.getElementById("uncertainty-config").classList.toggle("hidden", !selected.includes("calibration"));
}

function renderEvaluatorOptions() {
    const model = pluginById("models", document.getElementById("model_type").value);
    const dataset = pluginById("datasets", document.getElementById("dataset").value);
    const container = document.getElementById("evaluator-options");
    const unavailable = [];
    container.textContent = "";
    pluginCatalog.evaluators.forEach(evaluator => {
        const compatibility = evaluatorCompatibility(evaluator, model, dataset);
        const option = document.createElement("label");
        option.className = `evaluator-option${compatibility.compatible ? "" : " incompatible"}`;
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.dataset.evaluatorId = evaluator.id;
        checkbox.checked = compatibility.compatible;
        checkbox.disabled = !compatibility.compatible;
        checkbox.addEventListener("change", renderEvaluatorParameters);
        const copy = document.createElement("span");
        const title = document.createElement("strong");
        title.textContent = evaluator.name;
        const description = document.createElement("small");
        description.textContent = evaluator.description;
        copy.append(title, description);
        option.append(checkbox, copy);
        container.appendChild(option);
        if (!compatibility.compatible) unavailable.push(evaluator.name);
    });
    document.getElementById("evaluator-compatibility-message").textContent = unavailable.length
        ? `Unavailable for this combination: ${unavailable.join(", ")}.`
        : "All installed benchmarks are compatible with this combination.";
    renderEvaluatorParameters();
}

function populatePluginSelect(select, plugins, preferred) {
    select.textContent = "";
    plugins.forEach(plugin => {
        const option = document.createElement("option");
        option.value = plugin.id;
        option.textContent = plugin.name;
        select.appendChild(option);
    });
    if (plugins.some(plugin => plugin.id === preferred)) select.value = preferred;
}

async function loadPluginCatalog() {
    const response = await fetch("/api/plugins");
    if (!response.ok) throw new Error("Could not load installed plugins");
    pluginCatalog = await response.json();
    Object.entries(pluginCatalog.core_parameters || {}).forEach(([name, schema]) => {
        const input = document.getElementById(name);
        if (!input || input.type !== "number") return;
        input.min = schema.minimum;
        input.max = schema.maximum;
        input.step = schema.type === "integer" ? "1" : "any";
        if (schema.default !== undefined) input.value = schema.default;
    });
    populatePluginSelect(document.getElementById("model_type"), pluginCatalog.models, "TransE");
    populatePluginSelect(document.getElementById("dataset"), pluginCatalog.datasets, "Synthetic");
    populatePluginSelect(document.getElementById("filter-model"), [{id: "ALL", name: "All models"}, ...pluginCatalog.models], "ALL");
    populatePluginSelect(document.getElementById("filter-dataset"), [{id: "ALL", name: "All datasets"}, ...pluginCatalog.datasets], "ALL");
    renderPluginParameters("models");
    renderPluginParameters("datasets");
    renderEvaluatorOptions();
}

function setFieldError(input, message) {
    const holder = input.closest(".param-card, .form-group");
    if (!holder) return;
    let note = holder.querySelector(".field-error");
    if (message) {
        if (!note) {
            note = document.createElement("small");
            note.className = "field-error";
            holder.appendChild(note);
        }
        note.textContent = message;
        input.setAttribute("aria-invalid", "true");
    } else {
        note?.remove();
        input.removeAttribute("aria-invalid");
    }
}

function validateField(input) {
    if (input.disabled || input.closest(".hidden")) {
        setFieldError(input, "");
        return true;
    }
    const raw = input.value.trim();
    const value = Number(raw);
    const lower = Number(input.min);
    const upper = Number(input.max);
    let message = "";
    if (!raw || !Number.isFinite(value)) message = "Enter a number.";
    else if (input.step === "1" && !Number.isInteger(value)) message = "Enter a whole number.";
    else if (input.min !== "" && value < lower || input.max !== "" && value > upper) {
        message = input.min !== "" && input.max !== ""
            ? `Use a value from ${input.min} to ${input.max}.`
            : input.min !== "" ? `Use at least ${input.min}.` : `Use at most ${input.max}.`;
    }
    setFieldError(input, message);
    return !message;
}

function syncEvaluationControls() {
    const full = document.getElementById("full_evaluation").checked;
    ["validation_fraction", "validation_sample_size", "test_fraction", "test_sample_size"].forEach(id => {
        const input = document.getElementById(id);
        input.disabled = full;
        input.closest(".param-card").classList.toggle("parameter-inactive", full);
        if (full) setFieldError(input, "");
    });
}

function validateForm() {
    const inputs = [...document.querySelectorAll("#benchmark-form input[type=number]")]
        .filter(input => input.type !== "hidden");
    let firstInvalid = null;
    for (const input of inputs) {
        if (!validateField(input) && !firstInvalid) firstInvalid = input;
    }
    const positive = document.getElementById("soft_label_pos");
    const negative = document.getElementById("soft_label_neg");
    if (document.getElementById("uncertainty_level").value === "custom"
        && !firstInvalid && Number(positive.value) < Number(negative.value)) {
        setFieldError(positive, "Positive label must be at least the negative label.");
        firstInvalid = positive;
    }
    if (firstInvalid) {
        firstInvalid.closest(".advanced-settings")?.setAttribute("open", "");
        firstInvalid.focus();
        document.getElementById("form-error").textContent = "Correct the highlighted parameter before starting the benchmark.";
        document.getElementById("form-error").classList.remove("hidden");
    }
    return !firstInvalid;
}

function collectConfig() {
    const fullEvaluation = document.getElementById("full_evaluation").checked;
    const customUncertainty = document.getElementById("uncertainty_level").value === "custom";
    return collectPluginParameters({
        experiment_name: document.getElementById("experiment_name").value.trim(),
        model_type: document.getElementById("model_type").value,
        dataset: document.getElementById("dataset").value,
        evaluators: [...document.querySelectorAll("[data-evaluator-id]:checked")].map(input => input.dataset.evaluatorId),
        seed: numberValue("seed"),
        embedding_dimension: numberValue("embedding_dimension"),
        learning_rate: numberValue("learning_rate"),
        batch_size: numberValue("batch_size"),
        epochs: numberValue("epochs"),
        margin: numberValue("margin"),
        p_norm: numberValue("p_norm"),
        train_fraction: numberValue("train_fraction"),
        validation_fraction: fullEvaluation ? 1 : numberValue("validation_fraction"),
        test_fraction: fullEvaluation ? 1 : numberValue("test_fraction"),
        train_sample_size: numberValue("train_sample_size"),
        eval_sample_size: numberValue("eval_sample_size"),
        validation_sample_size: fullEvaluation ? 0 : numberValue("validation_sample_size"),
        test_sample_size: fullEvaluation ? 0 : numberValue("test_sample_size"),
        full_evaluation: fullEvaluation,
        uncertainty_level: document.getElementById("uncertainty_level").value,
        soft_label_pos: customUncertainty ? numberValue("soft_label_pos") : 0.95,
        soft_label_neg: customUncertainty ? numberValue("soft_label_neg") : 0.05,
        noise_level: customUncertainty ? numberValue("noise_level") : 0,
        alpha: numberValue("alpha")
    });
}

async function submitRun(event) {
    event.preventDefault();
    if (!validateForm()) return;
    const button = document.getElementById("run-button");
    const errorBox = document.getElementById("form-error");
    errorBox.classList.add("hidden");
    button.disabled = true;
    button.textContent = "Starting…";
    try {
        const response = await fetch("/api/runs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(collectConfig())});
        const data = await response.json();
        if (!response.ok) {
            if (data.field) {
                const field = document.getElementById(data.field)
                    || document.querySelector(`[data-plugin-param="${data.field}"]`);
                if (field) {
                    setFieldError(field, data.error);
                    field.closest(".advanced-settings")?.setAttribute("open", "");
                    field.focus();
                }
            }
            throw new Error(data.error || "Could not start the benchmark");
        }
        watchRun(data.id);
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        button.disabled = false;
        button.textContent = "Run real benchmark";
    }
}

function watchRun(runId) {
    if (activeEvents) activeEvents.close();
    const panel = document.getElementById("console-panel");
    const logs = document.getElementById("console-logs");
    panel.classList.remove("hidden");
    logs.textContent = "";
    document.getElementById("run-status").textContent = "running";
    document.getElementById("console-title").textContent = `Executing run #${runId}`;
    activeEvents = new EventSource(`/api/runs/${runId}/events`);
    activeEvents.onmessage = async event => {
        if (event.data.startsWith("[FINISHED]")) {
            activeEvents.close();
            activeEvents = null;
            const completed = event.data.includes("completed");
            document.getElementById("run-status").textContent = completed ? "completed" : "failed";
            const button = document.getElementById("run-button");
            button.disabled = false;
            button.textContent = "Run real benchmark";
            await loadRuns();
            const runWorkspaceVisible = document.getElementById("run-section").classList.contains("active");
            if (runWorkspaceVisible) {
                await showRun(runId);
            } else if (document.getElementById("tracking-section").classList.contains("active")) {
                showTrackingMessage(`Run #${runId} ${completed ? "completed" : "failed"}. The tracking table is up to date.`, !completed);
            }
            return;
        }
        const line = document.createElement("div");
        line.className = "console-log-line";
        if (event.data.startsWith("Epoch")) line.classList.add("epoch");
        if (event.data.startsWith("METRIC")) line.classList.add("loss");
        line.textContent = event.data;
        logs.appendChild(line);
        logs.scrollTop = logs.scrollHeight;
    };
    activeEvents.onerror = () => {
        const line = document.createElement("div");
        line.className = "console-log-line error";
        line.textContent = "Live log connection interrupted. The run may still be executing; check Tracking.";
        logs.appendChild(line);
    };
}

function appendDefinition(list, label, value, title = "") {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value ?? "—";
    if (title) description.title = title;
    row.append(term, description);
    list.appendChild(row);
}

async function showRun(runId) {
    const response = await fetch(`/api/runs/${runId}`);
    if (!response.ok) return;
    const run = await response.json();
    document.getElementById("no-results-view").classList.add("hidden");
    document.getElementById("results-view").classList.remove("hidden");
    document.getElementById("result-id").textContent = `RUN #${run.id}`;
    document.getElementById("result-name").textContent = run.experiment_name;
    document.getElementById("result-subtitle").textContent = `${run.model_type} · ${run.dataset} · seed ${run.seed}`;
    document.getElementById("result-status").textContent = run.status;
    document.getElementById("result-status").className = `badge ${run.status === "completed" ? "badge-success" : run.status === "failed" ? "badge-danger" : "badge-warning"}`;

    const metricGrid = document.getElementById("metric-grid");
    metricGrid.textContent = "";
    metricDefinitionsForRun(run).forEach(([key, label, direction]) => {
        const card = document.createElement("div");
        card.className = "kpi-card metric-card";
        const labelElement = document.createElement("span");
        labelElement.className = "kpi-label";
        labelElement.textContent = label;
        const valueElement = document.createElement("span");
        valueElement.className = "kpi-value";
        valueElement.textContent = formatMetric(run[key], key === "mean_rank" ? 2 : 4);
        const hint = document.createElement("span");
        hint.className = "kpi-subtext";
        hint.textContent = `${direction} is better`;
        card.append(labelElement, valueElement, hint);
        metricGrid.appendChild(card);
    });

    const execution = document.getElementById("execution-metadata");
    execution.textContent = "";
    appendDefinition(execution, "Model", run.model_type);
    appendDefinition(execution, "Model plugin", run.model_plugin_version || "Legacy run");
    appendDefinition(execution, "Dataset", run.dataset);
    appendDefinition(execution, "Dataset plugin", run.dataset_plugin_version || "Legacy run");
    appendDefinition(execution, "Evaluation protocol", run.evaluation_protocol_version || "Legacy run");
    appendDefinition(execution, "Evaluators", (run.evaluators || []).map(item => `${item.id}@${item.version}`).join(", ") || "Legacy built-in evaluation");
    appendDefinition(execution, "Evaluation", run.full_evaluation ? "Full test split" : `Sample of ${run.eval_sample_size}`);
    appendDefinition(execution, "Uncertainty protocol", run.uncertainty_protocol);
    appendDefinition(execution, "Device", run.device);
    appendDefinition(execution, "Training time", run.train_seconds === null ? "—" : `${formatMetric(run.train_seconds, 2)} s`);
    appendDefinition(execution, "Evaluation time", run.eval_seconds === null ? "—" : `${formatMetric(run.eval_seconds, 2)} s`);
    appendDefinition(execution, "Created", run.created_at);
    if (run.error_message) appendDefinition(execution, "Error", run.error_message);

    const fingerprints = document.getElementById("fingerprint-metadata");
    fingerprints.textContent = "";
    appendDefinition(fingerprints, "Configuration", shortHash(run.config_hash), run.config_hash);
    appendDefinition(fingerprints, "Thesis code", shortHash(run.code_fingerprint), run.code_fingerprint);
    appendDefinition(fingerprints, "Dataset", shortHash(run.dataset_fingerprint), run.dataset_fingerprint);
    appendDefinition(fingerprints, "Artifact directory", run.artifact_dir, run.artifact_dir);
    switchSection("results-section");
}

function createCell(text, className = "") {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (className) cell.className = className;
    return cell;
}

function filteredRuns() {
    const dataset = document.getElementById("filter-dataset").value;
    const model = document.getElementById("filter-model").value;
    return runCache.filter(run => (dataset === "ALL" || run.dataset === dataset) && (model === "ALL" || run.model_type === model));
}

function populateRuns() {
    const selectedBeforeRefresh = new Set(selectedRunIds());
    const body = document.getElementById("run-table-body");
    body.textContent = "";
    filteredRuns().forEach(run => {
        const row = document.createElement("tr");
        const selection = document.createElement("td");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "run-selector";
        checkbox.value = run.id;
        checkbox.disabled = run.status === "queued" || run.status === "running";
        checkbox.setAttribute("aria-label", `${checkbox.disabled ? "Run" : "Select run"} #${run.id}: ${run.experiment_name}`);
        checkbox.checked = !checkbox.disabled && selectedBeforeRefresh.has(run.id);
        checkbox.addEventListener("change", updateSelectionActions);
        selection.appendChild(checkbox);
        row.append(selection);
        row.append(createCell(`#${run.id}`, "mono"), createCell(run.created_at), createCell(run.experiment_name), createCell(run.model_type), createCell(run.dataset), createCell(run.seed), createCell(run.full_evaluation ? "full" : `sample ${run.eval_sample_size}`), createCell(formatMetric(run.mrr)), createCell(formatMetric(run.hits_at_10)), createCell(formatMetric(run.ece)), createCell(formatMetric(run.f1_score)), createCell(run.status));
        const action = document.createElement("td");
        const button = document.createElement("button");
        button.className = "btn btn-primary btn-sm";
        button.textContent = "View";
        button.setAttribute("aria-label", `View run #${run.id}: ${run.experiment_name}`);
        button.addEventListener("click", () => showRun(run.id));
        action.appendChild(button);
        row.appendChild(action);
        body.appendChild(row);
    });
    updateSelectionActions();
}

function selectedRunIds() {
    return [...document.querySelectorAll(".run-selector:checked")].map(element => Number(element.value));
}

function updateSelectionActions() {
    const count = selectedRunIds().length;
    const deleteButton = document.getElementById("delete-selected-button");
    const compareButton = document.getElementById("compare-button");
    deleteButton.disabled = count === 0;
    deleteButton.textContent = count ? `Delete selected (${count})` : "Delete selected";
    compareButton.disabled = count < 2 || count > 4;
    compareButton.textContent = count ? `Compare selected (${count}/4)` : "Compare selected";
}

function showTrackingMessage(message, isError = false) {
    const box = document.getElementById("tracking-message");
    box.textContent = message;
    box.className = `tracking-message ${isError ? "error" : "success"}`;
}

function openDeleteModal(mode) {
    const ids = selectedRunIds();
    const count = mode === "all" ? runCache.length : ids.length;
    if (!count) {
        showTrackingMessage(mode === "all" ? "There are no runs to delete." : "Select at least one run to delete.", true);
        return;
    }
    pendingDeleteMode = mode;
    document.getElementById("delete-modal-title").textContent = mode === "all" ? "Delete all experiment runs?" : `Delete ${count} selected run${count === 1 ? "" : "s"}?`;
    document.getElementById("delete-modal-copy").textContent = mode === "all"
        ? `This removes all ${count} database records and their stored run artifacts.`
        : `This removes ${count} selected database record${count === 1 ? "" : "s"} and the associated run artifacts.`;
    document.getElementById("delete-confirm-button").textContent = mode === "all" ? "Delete all runs" : "Delete selected runs";
    document.getElementById("delete-modal-error").classList.add("hidden");
    document.getElementById("delete-modal").classList.remove("hidden");
    document.getElementById("delete-cancel-button").focus();
}

function closeDeleteModal() {
    pendingDeleteMode = null;
    document.getElementById("delete-modal").classList.add("hidden");
}

async function confirmRunDeletion() {
    if (!pendingDeleteMode) return;
    const mode = pendingDeleteMode;
    const button = document.getElementById("delete-confirm-button");
    const errorBox = document.getElementById("delete-modal-error");
    button.disabled = true;
    button.textContent = "Deleting…";
    errorBox.classList.add("hidden");
    try {
        const payload = mode === "all"
            ? {all: true, confirmation: "DELETE ALL"}
            : {ids: selectedRunIds(), confirmation: "DELETE SELECTED"};
        const response = await fetch("/api/runs", {
            method: "DELETE",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not delete the runs");
        closeDeleteModal();
        await loadRuns();
        const warning = data.warnings?.length ? ` ${data.warnings.join(" ")}` : "";
        showTrackingMessage(`${data.deleted} run${data.deleted === 1 ? "" : "s"} deleted.${warning}`);
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
    } finally {
        button.disabled = false;
        if (pendingDeleteMode) button.textContent = mode === "all" ? "Delete all runs" : "Delete selected runs";
    }
}

function renderLandingDashboard() {
    const completed = runCache.filter(run => run.status === "completed");
    const latest = runCache[0];
    const best = completed.reduce((current, run) => {
        if (run.mrr === null || run.mrr === undefined) return current;
        return !current || Number(run.mrr) > Number(current.mrr) ? run : current;
    }, null);

    document.getElementById("landing-total-runs").textContent = runCache.length;
    document.getElementById("landing-completed-runs").textContent = completed.length;
    document.getElementById("landing-run-trend").textContent = runCache.length
        ? `${runCache.filter(run => run.status === "running" || run.status === "queued").length} currently active`
        : "Ready for your first benchmark";
    document.getElementById("landing-completion-rate").textContent = runCache.length
        ? `${Math.round((completed.length / runCache.length) * 100)}% completion rate`
        : "No completed runs yet";
    document.getElementById("landing-best-mrr").textContent = best ? formatMetric(best.mrr, 3) : "—";
    document.getElementById("landing-best-model").textContent = best
        ? `${best.model_type} · ${best.dataset}`
        : "Awaiting evaluation";
    document.getElementById("landing-last-dataset").textContent = latest ? latest.dataset : "—";
    document.getElementById("landing-last-run").textContent = latest
        ? `Run #${latest.id} · ${latest.status}`
        : "No recent activity";

    const modelCounts = completed.reduce((counts, run) => {
        counts[run.model_type] = (counts[run.model_type] || 0) + 1;
        return counts;
    }, {});
    const modelEntries = Object.entries(modelCounts).sort((a, b) => b[1] - a[1]);
    const totalCoverage = completed.length;
    const colors = ["#2764e7", "#14a696", "#7c5ce7", "#e08a2e", "#d84e78", "#56728f"];
    const ring = document.getElementById("landing-coverage-ring");
    let cursor = 0;
    const segments = modelEntries.map(([, count], index) => {
        const start = cursor;
        cursor += (count / totalCoverage) * 100;
        return `${colors[index % colors.length]} ${start}% ${cursor}%`;
    });
    ring.style.background = totalCoverage ? `conic-gradient(${segments.join(", ")})` : "";
    ring.classList.toggle("empty", totalCoverage === 0);
    document.getElementById("landing-coverage-total").textContent = totalCoverage;
    const legend = document.getElementById("landing-model-legend");
    legend.textContent = "";
    if (!modelEntries.length) {
        const empty = document.createElement("div");
        empty.innerHTML = "<span class=\"legend-dot\"></span><span>No completed models</span><strong>0</strong>";
        legend.appendChild(empty);
    } else {
        modelEntries.forEach(([model, count], index) => {
            const item = document.createElement("div");
            const dot = document.createElement("span");
            dot.className = "legend-dot";
            dot.style.background = colors[index % colors.length];
            const label = document.createElement("span");
            label.textContent = pluginById("models", model)?.name || model;
            const value = document.createElement("strong");
            value.textContent = count;
            item.append(dot, label, value);
            legend.appendChild(item);
        });
    }

    const chart = document.getElementById("landing-run-chart");
    chart.textContent = "";
    const chartRuns = completed.filter(run => run.mrr !== null && run.mrr !== undefined).slice(0, 7).reverse();
    if (!chartRuns.length) {
        const empty = document.createElement("div");
        empty.className = "dashboard-empty";
        empty.innerHTML = "<span></span><strong>No performance data yet</strong><small>Complete a benchmark to build this chart.</small>";
        chart.appendChild(empty);
    } else {
        const maxMrr = Math.max(...chartRuns.map(run => Number(run.mrr)), 0.01);
        chartRuns.forEach(run => {
            const item = document.createElement("button");
            item.type = "button";
            item.className = "chart-column";
            item.title = `Run #${run.id}: ${formatMetric(run.mrr)} MRR`;
            item.setAttribute("aria-label", item.title);
            item.addEventListener("click", () => showRun(run.id));

            const value = document.createElement("span");
            value.className = "chart-value";
            value.textContent = formatMetric(run.mrr, 2);
            const track = document.createElement("span");
            track.className = "chart-track";
            const bar = document.createElement("span");
            bar.className = `chart-bar ${run.model_type === "FuzzyTransE" ? "fuzzy" : "transe"}`;
            bar.style.height = `${Math.max(12, (Number(run.mrr) / maxMrr) * 100)}%`;
            track.appendChild(bar);
            const label = document.createElement("span");
            label.className = "chart-label";
            label.textContent = `#${run.id}`;
            item.append(value, track, label);
            chart.appendChild(item);
        });
    }

    const recentList = document.getElementById("landing-recent-runs");
    recentList.textContent = "";
    if (!runCache.length) {
        const empty = document.createElement("div");
        empty.className = "recent-empty";
        empty.textContent = "Your first experiment will appear here.";
        recentList.appendChild(empty);
    } else {
        runCache.slice(0, 4).forEach(run => {
            const item = document.createElement("button");
            item.type = "button";
            item.className = "recent-run";
            item.addEventListener("click", () => showRun(run.id));

            const identity = document.createElement("span");
            identity.className = "recent-identity";
            const icon = document.createElement("i");
            icon.textContent = run.model_type === "FuzzyTransE" ? "F" : "T";
            icon.className = run.model_type === "FuzzyTransE" ? "fuzzy" : "transe";
            const copy = document.createElement("span");
            const name = document.createElement("strong");
            name.textContent = run.experiment_name;
            const meta = document.createElement("small");
            meta.textContent = `#${run.id} · ${run.model_type} · ${run.dataset}`;
            copy.append(name, meta);
            identity.append(icon, copy);

            const result = document.createElement("span");
            result.className = "recent-result";
            const metric = document.createElement("strong");
            metric.textContent = run.status === "completed" ? formatMetric(run.mrr, 3) : "—";
            const status = document.createElement("small");
            status.textContent = run.status === "completed" ? "MRR" : run.status;
            status.className = `recent-status ${run.status}`;
            result.append(metric, status);
            item.append(identity, result);
            recentList.appendChild(item);
        });
    }
}

async function loadRuns() {
    if (runsRequestInFlight) return;
    runsRequestInFlight = true;
    try {
        const response = await fetch("/api/runs", {cache: "no-store"});
        if (!response.ok) return;
        runCache = await response.json();
        populateRuns();
        renderLandingDashboard();
    } finally {
        runsRequestInFlight = false;
        updateTrackingRefresh();
    }
}

function compareSelected() {
    const ids = selectedRunIds();
    const selected = runCache.filter(run => ids.includes(run.id));
    const panel = document.getElementById("comparison-panel");
    panel.textContent = "";
    if (selected.length < 2 || selected.length > 4) {
        panel.textContent = "Select between two and four completed runs to open the visual comparison page.";
        panel.classList.remove("hidden");
        return;
    }
    if (selected.some(run => run.status !== "completed")) {
        panel.textContent = "Only completed runs can be compared. Failed runs may be selected for deletion only.";
        panel.classList.remove("hidden");
        return;
    }
    window.location.href = `/compare?ids=${selected.map(run => run.id).join(",")}`;
}

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error();
        const health = await response.json();
        document.getElementById("health-label").textContent = `${health.installed_models} models · ${health.installed_datasets} datasets installed`;
    } catch (_) {
        document.getElementById("health-label").textContent = "Local service unavailable";
    }
}

document.querySelectorAll(".nav-item").forEach(button => {
    button.addEventListener("click", () => switchSection(button.dataset.section));
    button.addEventListener("keydown", event => {
        if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(event.key)) return;
        event.preventDefault();
        const tabs = [...document.querySelectorAll(".nav-item")];
        const offset = ["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1;
        const next = tabs[(tabs.indexOf(button) + offset + tabs.length) % tabs.length];
        switchSection(next.dataset.section);
        next.focus();
    });
});
document.getElementById("model_type").addEventListener("change", () => {
    renderPluginParameters("models");
    renderEvaluatorOptions();
});
document.getElementById("dataset").addEventListener("change", () => {
    renderPluginParameters("datasets");
    renderEvaluatorOptions();
    if (document.getElementById("dataset").value === "Synthetic") {
        document.getElementById("batch_size").value = pluginCatalog.core_parameters.batch_size.default;
        document.getElementById("train_sample_size").value = 0;
        document.getElementById("eval_sample_size").value = 0;
    }
});
document.getElementById("uncertainty_level").addEventListener("change", updateConditionalFields);
document.getElementById("full_evaluation").addEventListener("change", syncEvaluationControls);
document.getElementById("benchmark-form").noValidate = true;
document.getElementById("benchmark-form").addEventListener("focusout", event => {
    if (event.target.matches('input[type="number"]')) validateField(event.target);
});
document.getElementById("benchmark-form").addEventListener("input", event => {
    if (event.target.matches('input[type="number"][aria-invalid="true"]')) validateField(event.target);
});
function stepLearningRate(direction) {
    const input = document.getElementById("learning_rate");
    const current = Number(input.value);
    if (!Number.isFinite(current)) return;
    const next = Math.round((current + direction * 0.01) * 100000000) / 100000000;
    if (next >= Number(input.min) && next <= Number(input.max)) input.value = String(next);
    validateField(input);
}
document.getElementById("learning_rate").addEventListener("keydown", event => {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    stepLearningRate(event.key === "ArrowUp" ? 1 : -1);
});
document.querySelectorAll("[data-learning-step]").forEach(button => button.addEventListener("click", () => {
    stepLearningRate(button.dataset.learningStep === "up" ? 1 : -1);
}));
syncEvaluationControls();
document.getElementById("benchmark-form").addEventListener("submit", submitRun);
document.getElementById("filter-dataset").addEventListener("change", populateRuns);
document.getElementById("filter-model").addEventListener("change", populateRuns);
document.getElementById("compare-button").addEventListener("click", compareSelected);
document.getElementById("delete-selected-button").addEventListener("click", () => openDeleteModal("selected"));
document.getElementById("delete-all-button").addEventListener("click", () => openDeleteModal("all"));
document.getElementById("delete-cancel-button").addEventListener("click", closeDeleteModal);
document.getElementById("delete-modal-backdrop").addEventListener("click", closeDeleteModal);
document.getElementById("delete-confirm-button").addEventListener("click", confirmRunDeletion);
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !document.getElementById("delete-modal").classList.contains("hidden")) closeDeleteModal();
});
document.getElementById("hero-start").addEventListener("click", () => {
    document.getElementById("benchmark-workspace").scrollIntoView({behavior: "smooth", block: "start"});
});
document.getElementById("hero-tracking").addEventListener("click", () => switchSection("tracking-section"));
const overviewDashboard = document.querySelector(".research-overview");
if (overviewDashboard) {
    document.getElementById("tracking-section").prepend(overviewDashboard);
    overviewDashboard.classList.add("dashboard-ready");
}
const initialSection = document.body.dataset.initialSection || "run-section";
loadPluginCatalog().catch(error => {
    document.getElementById("form-error").textContent = error.message;
    document.getElementById("form-error").classList.remove("hidden");
});
updateConditionalFields();
checkHealth();
loadRuns();
switchSection(initialSection);
