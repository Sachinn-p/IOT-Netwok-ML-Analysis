const rowsKpi = document.getElementById("rows-kpi");
const colsKpi = document.getElementById("cols-kpi");
const bestModelKpi = document.getElementById("best-model-kpi");
const refreshKpi = document.getElementById("refresh-kpi");
const pipelineStatus = document.getElementById("pipeline-status");
const datasetPath = document.getElementById("dataset-path");
const systemState = document.getElementById("system-state");
const metricsTableBody = document.querySelector("#metrics-table tbody");
const predictForm = document.getElementById("predict-form");
const predictionResult = document.getElementById("prediction-result");
const trainButton = document.getElementById("train-btn");

function setPipelineStatus(message) {
  pipelineStatus.textContent = message;
}

function setSystemState(label, tone) {
  systemState.textContent = label;
  systemState.classList.remove("state-ok", "state-busy", "state-error");
  systemState.classList.add(tone);
}

function formatMatrix(matrix) {
  if (!Array.isArray(matrix)) return "-";
  return matrix.map((row) => `[${row.join(", ")}]`).join(" ");
}

function renderMetricsTable(metrics) {
  metricsTableBody.innerHTML = "";
  const modelNames = Object.keys(metrics || {});

  if (modelNames.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="3">No model metrics found. Run pipeline first.</td>`;
    metricsTableBody.appendChild(row);
    return;
  }

  modelNames.forEach((name) => {
    const data = metrics[name];
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${name}</td>
      <td>${Number(data.accuracy).toFixed(4)}</td>
      <td>${formatMatrix(data.confusion_matrix)}</td>
    `;
    metricsTableBody.appendChild(row);
  });
}

function refreshPlotImages() {
  const ts = Date.now();
  const ids = ["corr-plot", "bw-rate-plot", "dist-plot", "trend-plot"];
  ids.forEach((id) => {
    const image = document.getElementById(id);
    if (!image) return;
    const [baseUrl] = image.src.split("?");
    image.src = `${baseUrl}?t=${ts}`;
  });
}

async function loadSummary() {
  try {
    const response = await fetch("/api/summary");
    if (!response.ok) throw new Error(`Summary API failed (${response.status})`);
    const data = await response.json();

    rowsKpi.textContent = data.dataset.shape.rows.toLocaleString();
    colsKpi.textContent = data.dataset.shape.columns.toLocaleString();
    bestModelKpi.textContent = data.best_model || "-";
    refreshKpi.textContent = new Date(data.generated_at).toLocaleTimeString();
    datasetPath.textContent = data.dataset.dataset_path || "Unavailable";

    renderMetricsTable(data.model_metrics || {});
    setPipelineStatus("Summary loaded. Use 'Run Full Pipeline' to retrain and regenerate plots.");
    setSystemState("Online", "state-ok");
  } catch (error) {
    setPipelineStatus(`Failed to load summary: ${error.message}`);
    setSystemState("Offline", "state-error");
  }
}

async function runTraining() {
  setPipelineStatus("Pipeline running... This may take up to 1-2 minutes.");
  setSystemState("Training", "state-busy");
  trainButton.disabled = true;
  trainButton.textContent = "Running...";

  try {
    const response = await fetch("/api/train", { method: "POST" });
    if (!response.ok) throw new Error(`Training failed (${response.status})`);
    const data = await response.json();

    bestModelKpi.textContent = data.best_model;
    renderMetricsTable(data.model_metrics || {});
    refreshPlotImages();

    const statusMessage = [
      data.message,
      `Best model: ${data.best_model}`,
      `Rows x Cols after normalization: ${data.normalized_shape.rows} x ${data.normalized_shape.columns}`,
      `Congestion: ${JSON.stringify(data.congestion_distribution)}`,
    ].join("\n");
    setPipelineStatus(statusMessage);
    setSystemState("Online", "state-ok");
  } catch (error) {
    const hint =
      error && String(error.message).toLowerCase().includes("network")
        ? "\nHint: server may have restarted or is unreachable. Run without --reload for training."
        : "";
    setPipelineStatus(`Training error: ${error.message}${hint}`);
    setSystemState("Fault", "state-error");
  } finally {
    trainButton.disabled = false;
    trainButton.textContent = "Run Full Pipeline";
  }
}

async function runPrediction(event) {
  event.preventDefault();

  const payload = {
    packet_rate: Number(document.getElementById("packet_rate").value),
    bandwidth: Number(document.getElementById("bandwidth").value),
    latency: Number(document.getElementById("latency").value),
    packet_loss: Number(document.getElementById("packet_loss").value),
  };

  predictionResult.textContent = "Predicting...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }

    predictionResult.textContent = `Predicted Congestion: ${data.congestion_level} | Network Load: ${data.network_load.toFixed(
      2
    )}`;
    setSystemState("Online", "state-ok");
  } catch (error) {
    predictionResult.textContent = `Prediction failed: ${error.message}`;
    setSystemState("Fault", "state-error");
  }
}

trainButton.addEventListener("click", runTraining);
predictForm.addEventListener("submit", runPrediction);
loadSummary();
