/**
 * Interactive Plotly charts for the Divergence Tree presentation.
 *
 * Slide A  – Aggregate method comparison (bar chart + metric dropdown).
 * Slide B  – Factor breakdown (line chart + factor dropdown + metric dropdown).
 *
 * Data is loaded from window.CHART_DATA (set by chart-data.js).
 */
(function () {
  "use strict";

  var COLORS = {
    "\u03bb=0":            "#2E86AB",
    "\u03bb=2":            "#E76F51",
    "\u03bb=4":            "#2A9D8F",
    "\u03bb=8":            "#264653",
    "TwoStep (tuned)":     "#8338EC",
    "TwoStep (recall)":    "#E63946",
    "DivTree \u03bb=0":    "#2E86AB",
    "DivTree \u03bb=1":    "#F4A261",
    "DivTree \u03bb=2":    "#E76F51",
    "DivTree \u03bb=4":    "#2A9D8F",
    "DivTree \u03bb=8":    "#264653",
  };

  var LAYOUT_BASE = {
    paper_bgcolor: "rgba(255,255,255,0)",
    plot_bgcolor: "rgba(255,255,255,0.45)",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: "#1a1a2e" },
    margin: { t: 28, r: 20, b: 52, l: 56 },
    showlegend: true,
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "center", x: 0.5, font: { size: 11 } },
  };

  var METRIC_LABELS = {
    accuracy:           "Accuracy",
    recall_region_2:    "Recall (Region 2)",
    precision_region_2: "Precision (Region 2)",
    f1_region_2:        "F1 (Region 2)",
    fnr_region_2:       "FNR (Region 2)",
    n_leaves:           "Number of Leaves",
    cpu_time:           "CPU Time (s)",
  };

  var AGGREGATE_METRICS = [
    "accuracy", "recall_region_2", "precision_region_2",
    "f1_region_2", "fnr_region_2", "n_leaves", "cpu_time"
  ];

  var FACTOR_LABELS = {
    noise:     "Noise",
    data_size: "Sample Size",
    sparsity:  "Sparsity (k)",
    rareness:  "Rareness",
    intensity: "Intensity",
  };

  function getData() {
    return window.CHART_DATA || {};
  }

  // ---------------------------------------------------------------
  // Slide A: Aggregate comparison
  // ---------------------------------------------------------------

  function renderAggregate(metric) {
    var el = document.getElementById("plot-aggregate");
    var summaryData = getData().summary;
    if (!el || !summaryData) return;

    var metricData = summaryData.metrics[metric];
    if (!metricData) return;

    var labels = [];
    var means  = [];
    var stds   = [];
    var colors = [];

    Object.keys(metricData.series).forEach(function (name) {
      var s = metricData.series[name];
      if (s.mean == null) return;
      labels.push(name);
      means.push(s.mean);
      stds.push(s.std);
      colors.push(COLORS[name] || "#888");
    });

    var trace = {
      type: "bar",
      x: labels,
      y: means,
      error_y: { type: "data", array: stds, visible: true, thickness: 1.2, width: 4 },
      marker: { color: colors, line: { width: 0 } },
      hovertemplate: "%{x}<br>Mean: %{y:.4f}<extra></extra>",
    };

    var layout = Object.assign({}, LAYOUT_BASE, {
      showlegend: false,
      yaxis: {
        title: METRIC_LABELS[metric] || metric,
        gridcolor: "rgba(0,0,0,0.07)",
        zeroline: false,
      },
      xaxis: { tickangle: -25 },
    });

    Plotly.react(el, [trace], layout, { responsive: true, displayModeBar: false });
  }

  function initAggregate() {
    var select = document.getElementById("metric-select-aggregate");
    var summaryData = getData().summary;
    if (!select || !summaryData) return;

    select.innerHTML = "";
    AGGREGATE_METRICS.forEach(function (m) {
      if (!summaryData.metrics[m]) return;
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = METRIC_LABELS[m] || m;
      if (m === "accuracy") opt.selected = true;
      select.appendChild(opt);
    });

    renderAggregate("accuracy");

    select.addEventListener("change", function () {
      renderAggregate(this.value);
    });
  }

  // ---------------------------------------------------------------
  // Slide B: Factor breakdown
  // ---------------------------------------------------------------

  function renderFactor(factor, metric) {
    var el = document.getElementById("plot-factor");
    if (!el) return;

    var data = getData()[factor];
    if (!data) { el.textContent = "No data for factor: " + factor; return; }

    var metricData = data.metrics[metric];
    if (!metricData) { el.textContent = "Metric not available for this factor."; return; }

    var useLogX = !!data.logX;
    var traces = [];

    Object.keys(metricData.series).forEach(function (name) {
      var s = metricData.series[name];
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: name,
        x: metricData.x,
        y: s.mean,
        line: { color: COLORS[name] || "#888", width: 2.5 },
        marker: { size: 6, color: COLORS[name] || "#888" },
        hovertemplate: name + "<br>x=%{x}<br>y=%{y:.4f}<extra></extra>",
      });
    });

    var allY = [];
    traces.forEach(function (t) {
      t.y.forEach(function (v) { if (v != null && isFinite(v)) allY.push(v); });
    });
    var yMin = Math.min.apply(null, allY);
    var yMax = Math.max.apply(null, allY);
    var pad = (yMax - yMin) * 0.12 || 0.05;
    var lo = Math.max(0, yMin - pad);
    var hi = Math.min(1, yMax + pad);

    var layout = Object.assign({}, LAYOUT_BASE, {
      xaxis: {
        title: data.xLabel || factor,
        type: useLogX ? "log" : "linear",
        gridcolor: "rgba(0,0,0,0.07)",
      },
      yaxis: {
        title: METRIC_LABELS[metric] || metric,
        range: [lo, hi],
        gridcolor: "rgba(0,0,0,0.07)",
        zeroline: false,
      },
    });

    Plotly.react(el, traces, layout, { responsive: true, displayModeBar: false });
  }

  function initFactor() {
    var factorSel = document.getElementById("factor-select");
    var metricSel = document.getElementById("metric-select-factor");
    if (!factorSel || !metricSel) return;

    factorSel.innerHTML = "";
    Object.keys(FACTOR_LABELS).forEach(function (f) {
      var opt = document.createElement("option");
      opt.value = f;
      opt.textContent = FACTOR_LABELS[f];
      if (f === "noise") opt.selected = true;
      factorSel.appendChild(opt);
    });

    metricSel.innerHTML = "";
    ["accuracy", "recall_region_2", "precision_region_2"].forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = METRIC_LABELS[m] || m;
      if (m === "recall_region_2") opt.selected = true;
      metricSel.appendChild(opt);
    });

    function update() {
      renderFactor(factorSel.value, metricSel.value);
    }

    factorSel.addEventListener("change", update);
    metricSel.addEventListener("change", update);

    update();
  }

  // ---------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------

  function init() {
    if (!window.CHART_DATA) {
      console.warn("CHART_DATA not found. Make sure chart-data.js is loaded before charts.js.");
      return;
    }
    initAggregate();
    initFactor();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
