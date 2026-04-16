"use strict";

let selectedVehicle = null;

const fmt  = (v, dec = 2) => (v == null ? "—" : Number(v).toFixed(dec));
const fmtI = (v) => (v == null ? "—" : Math.round(v).toLocaleString());
const fmtPct = (v, dec = 2) => (v == null ? "—" : `${Number(v).toFixed(dec)}%`);

const PLOTLY_CFG = { displayModeBar: false, responsive: true };
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#f8fafc",
  font: { family: "Inter, system-ui, sans-serif", size: 11, color: "#475569" },
  margin: { l: 50, r: 20, t: 30, b: 40 },
};

// ── Composite score → inline left-border colour on first cell ────────────────
function scoreBorderColor(s) {
  if (s == null) return "#94a3b8";
  if (s >= 0.5)  return "#ef4444";
  if (s >= 0.35) return "#f59e0b";
  if (s >= 0.25) return "#3b82f6";
  return "#10b981";
}

// ── Reliability badge ─────────────────────────────────────────────────────────
function relBadge(r) {
  if (!r) return "—";
  const map = {
    reliable:          "badge-reliable",
    low_r2:            "badge-low-r2",
    insufficient_data: "badge-insuf",
  };
  return `<span class="badge ${map[r] || "bg-secondary"}">${r.replace(/_/g," ")}</span>`;
}

// ── Path badge ────────────────────────────────────────────────────────────────
function pathBadge(p) {
  if (!p) return "—";
  return `<span class="badge ${p === "cycle" ? "badge-cycle" : "badge-calendar"}">${p}</span>`;
}

// ── Overview + Health Summary table ──────────────────────────────────────────
async function loadOverview() {
  const d = await fetch("/api/overview/").then((r) => r.json());

  // KPI cards
  document.getElementById("s1Vehicles").textContent = d.n_vehicles;
  document.getElementById("s1MeanSoh").textContent  = fmtPct(d.fleet_mean_soh);
  document.getElementById("s1StdSoh").textContent   = fmtPct(d.fleet_std_soh, 3);
  document.getElementById("s1EkfRul").textContent   = d.median_ekf_rul  ? `${fmtI(d.median_ekf_rul)} days`  : "—";
  document.getElementById("s1Period").textContent   = `${d.first_date} → ${d.last_date}`;

  // Trend direction helper
  const trendVal = d.soh_trend_pct;
  const trendStr = trendVal != null
    ? `${trendVal > 0 ? "+" : ""}${trendVal.toFixed(2)}% over ${d.span_days} days`
    : "—";

  // Fleet Health Summary table
  document.getElementById("ht_meanSoh").textContent   = fmtPct(d.fleet_mean_soh, 3);
  document.getElementById("ht_stdSoh").textContent    = fmtPct(d.fleet_std_soh, 3);
  document.getElementById("ht_trend").textContent     = trendStr;
  document.getElementById("ht_cycleObs").textContent  = d.cycle_soh_obs_n
    ? `${d.cycle_soh_obs_n.toLocaleString()} (${d.cycle_soh_obs_pct}% of ${d.cycle_soh_total.toLocaleString()})`
    : "—";
  document.getElementById("ht_ekfRul").textContent    = d.median_ekf_rul  ? `${fmtI(d.median_ekf_rul)} days` : "—";
  document.getElementById("ht_eol").textContent       = d.eol_threshold   ? `${d.eol_threshold}% SoH` : "80% SoH";
  document.getElementById("ht_span").textContent      = `${d.first_date} → ${d.last_date} (${d.span_days} days)`;

}

// ── Quintile table ────────────────────────────────────────────────────────────
async function loadQuintiles() {
  const d = await fetch("/api/quintiles/").then((r) => r.json());
  document.querySelector("#quintileTable tbody").innerHTML = d.quintiles
    .map((r) => `<tr>
      <td>${r.quintile}</td>
      <td class="text-end fw-semibold">${fmt(r.median_soh, 3)}%</td>
    </tr>`)
    .join("");
}

// ── Per-vehicle table ─────────────────────────────────────────────────────────
async function loadVehicles() {
  const d = await fetch("/api/vehicles/").then((r) => r.json());
  const badge = document.getElementById("vehicleCount");
  if (badge) badge.textContent = `(${d.vehicles.length} vehicles)`;
  document.querySelector("#vehicleTable tbody").innerHTML = d.vehicles
    .map((v) => {
      const sc = v.composite_degradation_score;
      const bc  = scoreBorderColor(sc);
      return `<tr class="vehicle-row" onclick="selectVehicle('${v.registration_number}')">
        <td style="border-left:4px solid ${bc}"><code style="font-size:.8rem">${v.registration_number}</code></td>
        <td class="text-end">${fmtPct(v.current_soh)}</td>
        <td class="text-end">${fmt(v["soh_slope_%per_day"], 4)}</td>
        <td class="text-end">${fmtPct(v.bayes_soh_pred)}</td>
        <td class="text-end">${fmt(v.bayes_soh_std, 3)}</td>
        <td class="text-end fw-bold">${fmt(sc, 4)}</td>
        <td class="text-end">${v.n_combined_anom != null ? v.n_combined_anom : "—"}</td>
        <td>${relBadge(v.rul_reliability)}</td>
        <td>${pathBadge(v.dual_dominant_path)}</td>
      </tr>`;
    })
    .join("");
}

// ── Anomaly tiers ─────────────────────────────────────────────────────────────
async function loadAnomalyTiers() {
  const d = await fetch("/api/anomaly-tiers/").then((r) => r.json());

  document.querySelector("#tier1Table tbody").innerHTML = d.tier1.map((v) =>
    `<tr class="vehicle-row" onclick="selectVehicle('${v.registration_number}')">
      <td><code style="font-size:.8rem">${v.registration_number}</code></td>
      <td class="small">${v.primary_signal || ""}</td>
      <td class="text-end">${fmtPct(v.current_soh)}</td>
      <td class="text-end">${fmt(v.soh_slope, 4)}</td>
      <td class="text-end fw-bold">${fmt(v.composite, 4)}</td>
      <td class="text-end">${v.n_combined_anom}</td>
    </tr>`).join("");

  document.querySelector("#tier2Table tbody").innerHTML = d.tier2.map((v) =>
    `<tr class="vehicle-row" onclick="selectVehicle('${v.registration_number}')">
      <td><code style="font-size:.8rem">${v.registration_number}</code></td>
      <td class="small text-muted">${v.note || ""}</td>
      <td class="text-end">${fmtPct(v.current_soh)}</td>
      <td class="text-end">${fmt(v.soh_slope, 4)}</td>
      <td class="text-end fw-bold">${fmt(v.composite, 4)}</td>
      <td class="text-end">${v.n_combined_anom}</td>
    </tr>`).join("");

  document.querySelector("#tier3Table tbody").innerHTML = d.tier3.map((v) =>
    `<tr class="vehicle-row" onclick="selectVehicle('${v.registration_number}')">
      <td><code style="font-size:.8rem">${v.registration_number}</code></td>
      <td class="small text-muted">${v.note || ""}</td>
      <td class="text-end">${fmtPct(v.current_soh)}</td>
      <td class="text-end">${fmt(v.soh_slope, 4)}</td>
      <td class="text-end fw-bold">${fmt(v.composite, 4)}</td>
      <td class="text-end">${v.n_combined_anom}</td>
    </tr>`).join("");
}

// ── EKF SoH Bollinger Bands ───────────────────────────────────────────────────
async function loadSohBands(reg) {
  document.getElementById("sohBandsLabel").textContent  = reg;
  document.getElementById("sohBandsHeader").textContent = `EKF SoH ± 2σ — ${reg} (bands at charging · dots at discharge)`;
  const el = document.getElementById("sohBandsChart");
  el.innerHTML = `<div class="placeholder-msg"><div class="spinner-border spinner-border-sm text-primary"></div><p>Loading…</p></div>`;

  const d = await fetch(`/api/soh-bands/${reg}/`).then((r) => r.json());
  if (d.error) {
    el.innerHTML = `<div class="coef-warn m-3">${d.error}</div>`;
    return;
  }
  if (!d.bands || !d.bands.length) {
    el.innerHTML = `<div class="placeholder-msg"><div class="ico">📭</div><p>No EKF data for ${reg}</p></div>`;
    return;
  }

  if (typeof Plotly === "undefined") {
    el.innerHTML = `<div class="coef-warn m-3">Plotly not loaded — please refresh the page.</div>`;
    return;
  }

  try {
  const dates = d.bands.map((b) => b.date);
  const ekf   = d.bands.map((b) => b.ekf_soh);
  const upper = d.bands.map((b) => b.upper);
  const lower = d.bands.map((b) => b.lower);
  const bms   = d.bands.map((b) => b.bms_soh_obs);

  const traces = [
    // Band fill: lower baseline (invisible anchor)
    { x: dates, y: lower, type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip", name: "_lower" },
    // Upper band — fills down to lower
    { x: dates, y: upper, type: "scatter", mode: "lines",
      fill: "tonexty", fillcolor: "rgba(59,130,246,0.10)",
      line: { color: "rgba(99,102,241,0.45)", dash: "dot", width: 1 },
      name: "±2σ band", hovertemplate: "Upper: %{y:.2f}%<extra></extra>" },
    // EKF SoH center line
    { x: dates, y: ekf, type: "scatter", mode: "lines",
      line: { color: "#3b82f6", width: 2.5 },
      name: "EKF SoH", hovertemplate: "EKF SoH: %{y:.3f}%<extra></extra>" },
  ];

  // BMS SoH observations (raw, noisy) — overlay only, not used for y-axis range
  const bmsClean = bms.filter((v) => v != null && isFinite(v));
  if (bmsClean.length) {
    traces.push({
      x: dates, y: bms, type: "scatter", mode: "markers",
      marker: { color: "#f59e0b", size: 5, opacity: 0.65, symbol: "circle" },
      name: "BMS SoH (charging)", hovertemplate: "BMS: %{y:.1f}%<extra></extra>",
    });
  }

  // Discharge sessions — forward-filled EKF (no band, shows state between charges)
  if (d.discharge_ekf && d.discharge_ekf.length) {
    const discDates = d.discharge_ekf.map((b) => b.date);
    const discEkf   = d.discharge_ekf.map((b) => b.ekf_soh);
    traces.push({
      x: discDates, y: discEkf, type: "scatter", mode: "markers",
      marker: { color: "#94a3b8", size: 3, opacity: 0.5, symbol: "circle" },
      name: "EKF SoH (discharge, fwd-fill)",
      hovertemplate: "Discharge EKF: %{y:.3f}%<extra></extra>",
    });
  }

  // Y-axis range based on EKF bands only (BMS can have integer-step outliers)
  const bandY = [...ekf, ...upper, ...lower].filter((v) => v != null && isFinite(v));
  const yMin  = Math.floor(Math.min(...bandY) - 0.5);
  const yMax  = Math.ceil(Math.max(...bandY)  + 0.5);

  el.innerHTML = "";  // clear spinner before Plotly mounts
  Plotly.newPlot(
    "sohBandsChart",
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      margin: { l: 55, r: 20, t: 20, b: 55 },
      xaxis: { title: "Date", gridcolor: "#e2e8f0", tickangle: -30 },
      yaxis: { title: "SoH (%)", gridcolor: "#e2e8f0", range: [yMin, yMax] },
      legend: { orientation: "h", y: -0.22, font: { size: 11 } },
      hovermode: "x unified",
    },
    PLOTLY_CFG
  );
  } catch (err) {
    el.innerHTML = `<div class="coef-warn m-3">Chart error: ${err.message}</div>`;
  }
}

// ── Anomaly breakdown helpers ─────────────────────────────────────────────────
const DONUT_LAYOUT = {
  ...PLOTLY_LAYOUT_BASE,
  margin: { l: 10, r: 10, t: 10, b: 10 },
  showlegend: true,
  legend: { orientation: "h", x: 0.5, xanchor: "center", y: 0.04, font: { size: 10 }, tracegroupgap: 2 },
};
// Fixed-size config for donuts — prevents legend from shrinking the pie on the larger chart
const DONUT_W = 340, DONUT_H = 340;
const DONUT_CFG = { displayModeBar: false }; // no responsive — size is locked
const SIG_PALETTE = ["#ef4444","#f97316","#eab308","#10b981","#06b6d4","#8b5cf6","#3b82f6","#64748b"];

async function renderDetectorChart(byDetector) {
  document.getElementById("breakdownDetectorChart").innerHTML = "";
  await Plotly.newPlot(
    "breakdownDetectorChart",
    [{
      type: "pie", hole: 0.5,
      domain: { x: [0.05, 0.95], y: [0.30, 1] },
      labels: Object.keys(byDetector), values: Object.values(byDetector),
      marker: { colors: ["#3b82f6", "#f59e0b"] },
      textinfo: "percent", textposition: "inside",
      insidetextorientation: "radial",
      hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
    }],
    { ...DONUT_LAYOUT, width: DONUT_W, height: DONUT_H }, DONUT_CFG
  );
}

async function renderSignalChart(bySignal, scope, currentDetector) {
  document.getElementById("breakdownSignalChart").innerHTML = "";
  document.getElementById("breakdownSignalHeader").textContent = `Signal Breakdown — ${scope}`;
  const labels = Object.keys(bySignal);
  const values = Object.values(bySignal);
  if (!labels.length) {
    document.getElementById("breakdownSignalChart").innerHTML =
      `<div class="placeholder-msg"><div class="ico">📭</div><p>No signal data for ${scope}</p></div>`;
    return;
  }
  await Plotly.newPlot(
    "breakdownSignalChart",
    [{
      type: "pie", hole: 0.5,
      domain: { x: [0.05, 0.95], y: [0.30, 1] },
      labels, values,
      marker: { colors: SIG_PALETTE.slice(0, labels.length) },
      textinfo: "percent", textposition: "inside",
      insidetextorientation: "radial",
      hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
    }],
    { ...DONUT_LAYOUT, width: DONUT_W, height: DONUT_H }, DONUT_CFG
  );

  const el = document.getElementById("breakdownSignalChart");
  el.removeAllListeners("plotly_click");
  el.on("plotly_click", (data) => {
    const signal = data.points[0].label;
    _sessionFilter.signal = (_sessionFilter.signal === signal) ? null : signal;
    refreshSessionsTable();
    if (_sessionsCache) {
      document.getElementById("section8").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
}

async function renderSessionTypeChart(sessions, scope) {
  const hdr = document.getElementById("breakdownSessionTypeHeader");
  const el  = document.getElementById("breakdownSessionTypeChart");
  hdr.textContent = `Session Type — ${scope}`;
  if (!sessions || !sessions.length) {
    el.innerHTML = `<div class="placeholder-msg"><div class="ico">📭</div><p>No sessions for ${scope}</p></div>`;
    return;
  }
  const counts = {};
  sessions.forEach((s) => { const t = s.session_type || "unknown"; counts[t] = (counts[t] || 0) + 1; });
  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const TYPE_COLORS = { charging: "#f59e0b", discharge: "#3b82f6", idle: "#9ca3af" };
  const colors = labels.map((l) => TYPE_COLORS[l] || "#6b7280");

  el.innerHTML = "";
  // await so Plotly's event system is ready before we bind plotly_click
  await Plotly.newPlot(
    "breakdownSessionTypeChart",
    [{
      type: "pie", hole: 0.5,
      domain: { x: [0.05, 0.95], y: [0.30, 1] },
      labels, values,
      marker: { colors },
      textinfo: "percent", textposition: "inside",
      insidetextorientation: "radial",
      hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
    }],
    { ...DONUT_LAYOUT, width: DONUT_W, height: DONUT_H }, DONUT_CFG
  );

  el.removeAllListeners("plotly_click");
  el.on("plotly_click", _onSessionTypeClick);
}

async function _onSessionTypeClick(data) {
  const sessionType = data.points[0].label;
  _sessionFilter.sessionType = (_sessionFilter.sessionType === sessionType) ? null : sessionType;

  // Fetch signal breakdown filtered by session type (+ any active detector)
  const reg    = _sessionsCache ? document.getElementById("sessionVehicleLabel").textContent : null;
  const validReg = reg && reg !== "select a vehicle above";
  const base   = validReg ? `/api/anomaly-breakdown/${reg}/` : "/api/anomaly-breakdown/";
  const params = new URLSearchParams();
  if (_sessionFilter.detector)    params.set("detector",     _sessionFilter.detector);
  if (_sessionFilter.sessionType) params.set("session_type", _sessionFilter.sessionType);
  const url    = params.toString() ? base + "?" + params.toString() : base;
  const fd     = await fetch(url).then((r) => r.json());

  const scope = validReg ? reg : "Fleet-wide";
  const parts = [];
  if (_sessionFilter.detector)    parts.push(_sessionFilter.detector.toUpperCase());
  if (_sessionFilter.sessionType) parts.push(_sessionFilter.sessionType);
  await renderSignalChart(fd.by_signal, parts.length ? `${parts.join(" · ")} — ${scope}` : scope, _sessionFilter.detector);

  refreshSessionsTable();
  if (_sessionsCache) {
    document.getElementById("section8").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ── Anomaly breakdown donut charts ────────────────────────────────────────────
async function loadAnomalyBreakdown(reg = null) {
  if (typeof Plotly === "undefined") return;

  // All 3 charts are per-vehicle only — show placeholder until vehicle selected
  if (!reg) {
    const ph = `<div class="placeholder-msg" style="height:200px"><p style="color:#94a3b8;font-size:.85rem">Select a vehicle above to view breakdown</p></div>`;
    document.getElementById("breakdownDetectorHeader").textContent     = "Alert Source";
    document.getElementById("breakdownSignalHeader").textContent       = "Signal Breakdown";
    document.getElementById("breakdownSessionTypeHeader").textContent  = "Session Type";
    document.getElementById("breakdownDetectorChart").innerHTML    = ph;
    document.getElementById("breakdownSignalChart").innerHTML      = ph;
    document.getElementById("breakdownSessionTypeChart").innerHTML = ph;
    return;
  }

  const base  = `/api/anomaly-breakdown/${reg}/`;
  const d     = await fetch(base).then((r) => r.json());
  const scope = reg;

  document.getElementById("breakdownDetectorHeader").textContent = `Alert Source — ${scope}`;

  await renderDetectorChart(d.by_detector);
  await renderSignalChart(d.by_signal, scope, null);

  // Reset session filter when breakdown reloads
  _sessionFilter = { detector: null, signal: null, sessionType: null };
  refreshSessionsTable();

  // Click on left chart → update right chart + filter sessions by detector
  const el = document.getElementById("breakdownDetectorChart");
  el.removeAllListeners("plotly_click");
  el.on("plotly_click", async (data) => {
    const label    = data.points[0].label;
    const detector = label.toLowerCase().includes("isolation") ? "if" : "cusum";

    // Toggle detector filter
    _sessionFilter.detector = (_sessionFilter.detector === detector) ? null : detector;
    _sessionFilter.signal   = null;

    const activeDetector = _sessionFilter.detector;
    const params = new URLSearchParams();
    if (activeDetector)                params.set("detector",     activeDetector);
    if (_sessionFilter.sessionType)    params.set("session_type", _sessionFilter.sessionType);
    const url = params.toString() ? base + "?" + params.toString() : base;
    const fd  = await fetch(url).then((r) => r.json());
    const parts = [];
    if (activeDetector)                parts.push(label);
    if (_sessionFilter.sessionType)    parts.push(_sessionFilter.sessionType);
    const sigScope = parts.length ? `${parts.join(" · ")} — ${scope}` : scope;
    await renderSignalChart(fd.by_signal, sigScope, activeDetector);
    refreshSessionsTable();
    if (_sessionsCache) {
      document.getElementById("section8").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
}

// ── BayesianRidge coefficients ────────────────────────────────────────────────
async function loadBayesCoef(reg) {
  const url = reg ? `/api/bayes-coef/${reg}/` : "/api/bayes-coef/";
  const d = await fetch(url).then((r) => {
    if (!r.ok) return r.json().then((e) => ({ error: e.error }));
    return r.json();
  });

  if (d.error) {
    const msg = `<div class="coef-warn m-3">${d.error}</div>`;
    document.getElementById("coefGlobalChart").innerHTML  = msg;
    document.getElementById("coefVehicleChart").innerHTML = msg;
    return;
  }

  renderCoefChart("coefGlobalChart", d.global, "Fleet Global (median)", true);

  if (reg && Object.keys(d.vehicle).length > 0) {
    document.getElementById("coefVehicleHeader").textContent = reg;
    renderCoefChart("coefVehicleChart", d.vehicle, reg, true);
  } else if (reg) {
    document.getElementById("coefVehicleChart").innerHTML =
      `<div class="placeholder-msg"><div class="ico">📭</div><p>No coefficient data for ${reg}</p></div>`;
  }
}

function renderCoefChart(divId, coefObj, title, negativeOnly = false) {
  try {
    let entries = Object.entries(coefObj).sort((a, b) => a[1] - b[1]);
    if (negativeOnly) entries = entries.filter(([, v]) => v < 0);
    const labels  = entries.map(([k]) => k.replace(/_/g, " "));
    const values  = entries.map(([, v]) => v);
    const colors  = values.map(() => "#3b82f6");

    if (typeof Plotly === "undefined") {
      document.getElementById(divId).innerHTML =
        `<div class="coef-warn m-3">Plotly chart library not loaded. Check internet connection.</div>`;
      return;
    }

    const chartHeight = Math.max(260, entries.length * 26 + 80);
    document.getElementById(divId).innerHTML = "";
    document.getElementById(divId).style.minHeight = chartHeight + "px";
    Plotly.newPlot(
      divId,
      [{ type: "bar", orientation: "h", x: values, y: labels,
         marker: { color: colors }, hovertemplate: "%{y}: %{x:.6f}<extra></extra>" }],
      {
        ...PLOTLY_LAYOUT_BASE,
        height: chartHeight,
        margin: { l: 175, r: 20, t: 32, b: 45 },
        title:  { text: title, font: { size: 12, color: "#334155" } },
        xaxis:  { title: "Coefficient", gridcolor: "#e2e8f0", zerolinecolor: "#94a3b8" },
        yaxis:  { automargin: true, tickfont: { size: 10 } },
      },
      PLOTLY_CFG
    );
  } catch (e) {
    console.error("renderCoefChart error:", e);
    const el = document.getElementById(divId);
    if (el) el.innerHTML = `<div class="coef-warn m-3">Chart render error: ${e.message}</div>`;
  }
}

// ── Sessions — display columns (others are fetched for client-side filtering) ──
const SESSION_HEADERS = [
  "Vehicle","Start","End","Type","Start SoC","End SoC","EKF SoH","Duration (hrs)","Session Odo (km)",
  "IF Score","IF Anomaly","CUSUM Anomaly","Degr. Score","Reason","Flagged",
  "V-Sags","IR Mean","Spread","Energy/km","Energy KWh","Low SOC",
  "Ref Cap AH","Voltage","Current","Cap AH Dischrg.","Cap AH Chrg.","Cap AH Plugin",
  "Cycle SOH","Block Cap AH","Block Odm Km","Chg Rate KW",
  "Cell Spread","Weak Subsys.","Hot Subsys.","Subsys V Std",
  "Temp Rise","BMS Cov.","Speed","Is Loaded","Cum EFC",
  "Days Since First","Aging Index",
  "VSag Rate/hr","IR Event Rate","IR EWM10","Spread EWM10",
  "Temp EWM10","VSag EWM10","VSag Trend","IR Evt Trend",
  "IR Trend","Spread Trend","SoH Trend","C-Rate Chg",
  "DoD Stress","Thermal Stress","E/Loaded","Total Alerts",
  "Cell Health","Cell Undervolt","Cell Overvolt","Rapid Heat","High E/km","Slow Chg","Fast Chg",
];
const SESSION_FIELDS = [
  "registration_number","start_time_ist","end_time_ist","session_type","soc_start","soc_end","ekf_soh","duration_hr","odometer_km",
  "if_score","if_anomaly","cusum_anomaly","composite_degradation_score","anomaly_reason","is_anomalous",
  "n_vsag","ir_ohm_mean","cell_spread_mean","energy_per_km","energy_kwh","n_low_soc",
  "ref_capacity_ah","voltage_mean_new","current_mean_new","capacity_ah_discharge_new","capacity_ah_charge_new","capacity_ah_plugin_new",
  "cycle_soh","block_capacity_ah","block_odometer_km","charging_rate_kw",
  "cell_spread_max","weak_subsystem_consistency","hot_subsystem_consistency","subsystem_voltage_std",
  "temp_rise_rate","bms_coverage","speed_mean","is_loaded","cum_efc",
  "days_since_first","aging_index",
  "vsag_rate_per_hr","ir_event_rate","ir_ohm_mean_ewm10","cell_spread_mean_ewm10",
  "temp_rise_rate_ewm10","vsag_rate_per_hr_ewm10","vsag_trend_slope","ir_event_trend_slope",
  "ir_ohm_trend_slope","spread_trend_slope","soh_trend_slope","c_rate_chg",
  "dod_stress","thermal_stress","energy_per_loaded_session","total_alerts",
  "cell_health_poor","n_cell_undervoltage","n_cell_overvoltage","rapid_heating","high_energy_per_km","slow_charging","fast_charging",
];
const SESSION_BOOL    = new Set(["if_anomaly","cusum_anomaly","is_anomalous"]);

// Global sessions cache for client-side filtering
let _sessionsCache = null;   // { sessions, total_sessions, total_anomalous }
let _sessionFilter = { detector: null, signal: null, sessionType: null };

// ── Client-side signal filter maps ────────────────────────────────────────────
const CUSUM_SIGNAL_FILTER = {
  "EKF SoH Decline": (s) => !!(s.cusum_ekf_soh_alarm),
  "BMS SoH Decline": (s) => !!(s.cusum_soh_alarm),
  "Cycle SoH Drop":  (s) => !!(s.cusum_cycle_soh_alarm),
  "IR Degradation":  (s) => !!(s.cusum_ir_slope_alarm) || (s.n_high_ir > 0),
  "Cell Spread":     (s) => !!(s.cusum_spread_alarm) || !!(s.cusum_spread_slope_alarm),
  "Thermal Stress":  (s) => !!(s.cusum_heat_alarm),
  "Efficiency Loss": (s) => !!(s.cusum_epk_alarm),
  "Voltage Sag":     (s) => (s.n_vsag > 0),
};

const IF_SIGNAL_KEYWORDS = {
  "IR Degradation":          ["n_high_ir", "ir_ohm_mean", "d_n_high_ir", "ir_event_rate", "d_ir_ohm"],
  "Voltage Sag":             ["n_vsag", "d_vsag_per_cycle"],
  "Cell Spread / Imbalance": ["cell_spread", "n_cell_spread_warn", "subsystem_voltage_std"],
  "Thermal Stress":          ["temp_lowest_mean", "temp_max", "temp_rise_rate", "thermal_stress"],
  "Efficiency / Capacity":   ["energy_per_loaded_session", "capacity_ah_discharge"],
  "High DoD":                ["dod_stress"],
  "Low SoC / Undervoltage":  ["n_low_soc", "voltage_min"],
  "SoH Decline":             ["capacity_soh_disc_new", "soh_smooth", "ekf_soh_delta", "cycle_soh"],
  "Usage Pattern":           ["odometer_km", "duration_hr"],
};

function _reasonMatch(reason, keywords) {
  const r = (reason || "").toLowerCase();
  return keywords.some((kw) => r.includes(kw.toLowerCase()));
}

function _applySessionFilter(sessions, { excludeSessionType = false } = {}) {
  const { detector, signal, sessionType } = _sessionFilter;
  let filtered = sessions;

  if (detector === "if") {
    filtered = filtered.filter((s) => !!(s.if_anomaly));
  } else if (detector === "cusum") {
    filtered = filtered.filter((s) =>
      !!(s.cusum_ekf_soh_alarm) || !!(s.cusum_soh_alarm) || !!(s.cusum_cycle_soh_alarm) ||
      !!(s.cusum_heat_alarm)    || !!(s.cusum_spread_alarm) || !!(s.cusum_spread_slope_alarm) ||
      !!(s.cusum_epk_alarm)     || !!(s.cusum_ir_slope_alarm)
    );
  }

  if (signal) {
    if (detector === "if") {
      const keywords = IF_SIGNAL_KEYWORDS[signal] || [];
      filtered = filtered.filter((s) => _reasonMatch(s.if_reason, keywords));
    } else {
      const fn = CUSUM_SIGNAL_FILTER[signal];
      if (fn) filtered = filtered.filter(fn);
    }
  }

  if (!excludeSessionType && sessionType) {
    filtered = filtered.filter((s) => s.session_type === sessionType);
  }

  return filtered;
}

function _renderSessionRows(sessions, reg) {
  if (!sessions.length) return `<tr><td colspan="${SESSION_HEADERS.length}" class="text-center text-muted py-3">No sessions match the current filter.</td></tr>`;
  return sessions.map((s) => {
    const flagged = !!s.is_anomalous;
    const sid     = s.session_id ?? "";
    const cells   = SESSION_FIELDS.map((f) => {
      const v = s[f];
      if (SESSION_BOOL.has(f)) {
        return v ? `<td><span class="badge" style="background:#fef3c7;color:#92400e">Yes</span></td>` : `<td><span style="color:#cbd5e1;font-size:.75rem">—</span></td>`;
      }
      if (f === "start_time_ist" || f === "end_time_ist") return `<td style="white-space:nowrap">${v ?? "—"}</td>`;
      if (f === "registration_number") return `<td><code style="font-size:.78rem">${v ?? "—"}</code></td>`;
      if (f === "session_type") {
        const label = v === "charging" ? "Charging" : v === "discharge" ? "Discharging" : (v ?? "—");
        const color = v === "charging" ? "#fef3c7;color:#92400e" : "#eff6ff;color:#1d4ed8";
        return `<td><span class="badge" style="background:${color}">${label}</span></td>`;
      }
      if (f === "is_loaded") {
        if (v == null) return "<td>—</td>";
        return `<td>${v == 1 || v === true ? "Inbound" : "Outbound"}</td>`;
      }
      if (v == null) return "<td>—</td>";
      if (f === "soc_start" || f === "soc_end") return `<td class="text-end">${Math.round(v)}%</td>`;
      if (typeof v === "number") return `<td class="text-end">${fmt(v, 3)}</td>`;
      if (typeof v === "boolean") return `<td>${v ? "True" : "False"}</td>`;
      return `<td>${v}</td>`;
    }).join("");
    const rowStyle = flagged ? "background:#fffbeb" : "";
    const startLbl = (s.start_time_ist ?? "").replace(/'/g, "");
    const rowClick = sid ? `onclick="loadTelemetry('${reg}','${sid}','${s.session_type ?? ''}','${startLbl}')"` : "";
    const rowTitle = sid ? `title="Click to view raw telemetry for this session" style="${rowStyle};cursor:pointer"` : `style="${rowStyle}"`;
    return `<tr ${rowTitle} ${rowClick}>${cells}</tr>`;
  }).join("");
}

function refreshSessionsTable() {
  if (!_sessionsCache) return;
  const { detector, signal, sessionType } = _sessionFilter;

  // Sessions after detector+signal filter (but NOT sessionType) → drives the type chart
  const preType = _applySessionFilter(_sessionsCache.sessions, { excludeSessionType: true });
  const scope   = document.getElementById("sessionVehicleLabel").textContent;
  renderSessionTypeChart(preType, scope);

  // Final filtered list (all three filters applied)
  const filtered = sessionType ? preType.filter((s) => s.session_type === sessionType) : preType;

  let filterNote = "";
  if (detector)    filterNote += `${detector.toUpperCase()} only`;
  if (signal)      filterNote += (filterNote ? " · " : "") + signal;
  if (sessionType) filterNote += (filterNote ? " · " : "") + sessionType;
  if (filterNote)  filterNote = ` <span style="color:#3b82f6;font-weight:600">[${filterNote}]</span>`;

  const currentReg = document.getElementById("sessionVehicleLabel").textContent;
  const note = `Showing <strong>${filtered.length}</strong> anomalous sessions &nbsp;|&nbsp; ${_sessionsCache.total_anomalous} anomalous of ${_sessionsCache.total_sessions} total sessions for this vehicle${filterNote}`;

  document.getElementById("sessionsContainer").innerHTML = `
    <div style="overflow-x:auto;max-height:480px">
      <table class="table table-sm" style="font-size:.78rem">
        <thead><tr>${SESSION_HEADERS.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
        <tbody>${_renderSessionRows(filtered, currentReg)}</tbody>
      </table>
    </div>
    <div style="padding:6px 12px;font-size:.74rem;color:#94a3b8">${note} • amber = anomalous • click a row for raw telemetry • click charts above to filter</div>`;
}

async function loadSessions(reg) {
  document.getElementById("sessionVehicleLabel").textContent = reg;
  _sessionsCache  = null;
  _sessionFilter  = { detector: null, signal: null, sessionType: null };
  const container = document.getElementById("sessionsContainer");
  container.innerHTML = `<div class="placeholder-msg"><div class="spinner-border spinner-border-sm text-primary"></div><p>Loading sessions…</p></div>`;

  const d = await fetch(`/api/sessions/${reg}/`).then((r) => r.json());

  if (!d.sessions || d.sessions.length === 0) {
    container.innerHTML = `<div class="placeholder-msg"><div class="ico">✅</div><p>No anomalous sessions found for ${reg} (${d.total_sessions} total sessions, all clean)</p></div>`;
    return;
  }

  _sessionsCache = d;
  refreshSessionsTable();
}

// ── Session telemetry inline section ─────────────────────────────────────────
async function loadTelemetry(reg, sessionId, sessionType, startTime) {
  const section = document.getElementById("telemetrySection");
  const body    = document.getElementById("telemetryBody");
  const label   = document.getElementById("telemetryLabel");

  const typeLabel = sessionType === "charging" ? "Charging"
                  : sessionType === "discharge" ? "Discharge" : (sessionType ?? "");
  label.textContent = `${reg} · ${startTime || sessionId} (${typeLabel})`;

  body.innerHTML = `<div class="placeholder-msg"><div class="spinner-border spinner-border-sm text-primary"></div><p>Loading telemetry…</p></div>`;
  section.style.display = "";
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  const d = await fetch(`/api/telemetry/${reg}/${sessionId}/`).then((r) => r.json());
  if (d.error) {
    body.innerHTML = `<div class="coef-warn m-3">${d.error}</div>`;
    return;
  }
  if (!d.rows || !d.rows.length) {
    body.innerHTML = `<div class="placeholder-msg"><div class="ico">📭</div><p>No telemetry rows found for this session. Re-run data_prep_1.py to populate the telemetry DB.</p></div>`;
    return;
  }

  const rows = d.rows;
  const ts   = rows.map((r) => r.ts || r.gps_time);

  const TEL_LAYOUT = {
    ...PLOTLY_LAYOUT_BASE,
    margin: { l: 52, r: 12, t: 28, b: 45 },
    xaxis:  { title: { text: "Time", font: { size: 10 } }, gridcolor: "#e2e8f0", tickangle: -25, tickfont: { size: 9 } },
    yaxis:  { gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    showlegend: false,
  };

  // Chart definitions
  const isCharging = sessionType === "charging";
  const chartDefs = [
    { title: "SoC (%)",              fields: [{ f: "soc",          color: "#3b82f6", name: "SoC" }],          yLabel: "%" },
    { title: "Temperature (°C)",     fields: [{ f: "temperature_highest", color: "#ef4444", name: "Max" },
                                               { f: "temperature_lowest",  color: "#06b6d4", name: "Min" }],  yLabel: "°C", multi: true },
    { title: "Cell Spread (mV)",     fields: [{ f: "cell_spread",  color: "#f59e0b", name: "Spread" }],       yLabel: "mV" },
    { title: "IR (Ω)",               fields: [{ f: "ir_ohm",       color: "#8b5cf6", name: "IR" }],           yLabel: "Ω", connectgaps: true },
    { title: "Voltage Sag Flag",     fields: [{ f: "_vsag",        color: "#ef4444", name: "Sag" }],          yLabel: "flag", bar: true },
    { title: "Speed (km/h)",         fields: [{ f: "speed",        color: "#10b981", name: "Speed" }],        yLabel: "km/h" },
    { title: "Weak Subsystem #",     fields: [{ f: "min_cell_voltage_subsystem_number", color: "#f97316", name: "Weak Subsys" }], yLabel: "subsys ID" },
    { title: "Hot Subsystem #",      fields: [{ f: "temperature_highest_subsystem_number", color: "#ef4444", name: "Hot Subsys" }], yLabel: "subsys ID" },
    ...(isCharging ? [
      { title: "Charging Power (kW)", fields: [{ f: "_chg_pwr", color: "#0ea5e9", name: "Chg Power" }], yLabel: "kW" },
    ] : []),
  ];

  // Pre-compute charging power if needed
  let augRows = rows;
  if (isCharging) {
    augRows = rows.map((r) => ({
      ...r,
      _chg_pwr: (r.hves1_current != null && r.hves1_voltage_level != null)
        ? Math.abs(r.hves1_current * r.hves1_voltage_level) / 1000
        : null,
    }));
  }

  // Build HTML grid: 2 charts per row
  const chartIds = chartDefs.map((_, i) => `telChart_${i}`);
  const pairs = [];
  for (let i = 0; i < chartDefs.length; i += 2) {
    const left  = chartDefs[i];
    const right = chartDefs[i + 1];
    pairs.push(`
      <div class="row g-2 mb-2">
        <div class="col-md-6">
          <div class="panel"><div class="panel-hdr">${left.title}</div>
          <div id="${chartIds[i]}" style="height:200px"></div></div>
        </div>
        ${right ? `<div class="col-md-6">
          <div class="panel"><div class="panel-hdr">${right.title}</div>
          <div id="${chartIds[i+1]}" style="height:200px"></div></div>
        </div>` : ""}
      </div>`);
  }
  body.innerHTML = pairs.join("");

  // Render each chart
  const syncIds = [];  // line charts: trigger hover and receive crosshair
  const barIds  = [];  // bar charts: receive crosshair only
  chartDefs.forEach(({ fields, yLabel, multi, bar, connectgaps }, i) => {
    const id = chartIds[i];
    const traces = fields
      .filter(({ f }) => augRows.some((r) => r[f] != null))
      .map(({ f, color, name }) => {
        const t = {
          x: ts,
          y: augRows.map((r) => r[f] ?? null),
          type: bar ? "bar" : "scatter",
          name,
          hovertemplate: `%{x}<br>${name}: %{y:.3f}<extra></extra>`,
        };
        if (bar) { t.marker = { color }; }
        else     { t.mode = "lines"; t.line = { color, width: 1.5 }; if (connectgaps) t.connectgaps = true; }
        return t;
      });
    if (!traces.length) {
      document.getElementById(id).innerHTML =
        `<div class="placeholder-msg" style="height:100%"><p>No data</p></div>`;
      return;
    }
    const layout = {
      ...TEL_LAYOUT,
      showlegend: !!(multi),
      xaxis: { ...TEL_LAYOUT.xaxis, type: "date" },
      yaxis: { ...TEL_LAYOUT.yaxis, title: { text: yLabel, font: { size: 9 } } },
    };
    if (multi) layout.legend = { orientation: "h", x: 0.5, xanchor: "center", y: 1.12, font: { size: 9 } };
    Plotly.newPlot(id, traces, layout, PLOTLY_CFG);
    if (bar) barIds.push(id);
    else     syncIds.push(id);
  });

  // Synchronized crosshair across all charts (line charts trigger; bar charts receive)
  syncIds.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.removeAllListeners("plotly_hover");
    el.removeAllListeners("plotly_unhover");
    el.on("plotly_hover", (ev) => {
      const xv    = ev.points[0].x;
      const shape = [
        { type: "line", x0: xv, x1: xv, y0: 0, y1: 1, yref: "paper",
          line: { color: "#ef4444", width: 2, dash: "solid" } },
        { type: "line", x0: xv, x1: xv, y0: 0, y1: 1, yref: "paper",
          line: { color: "#fbbf24", width: 2, dash: "dot" } },
      ];
      [...syncIds.filter((i) => i !== id), ...barIds].forEach((oid) => {
        const oel = document.getElementById(oid);
        if (oel) Plotly.relayout(oel, { shapes: shape });
      });
    });
    el.on("plotly_unhover", () => {
      [...syncIds.filter((i) => i !== id), ...barIds].forEach((oid) => {
        const oel = document.getElementById(oid);
        if (oel) Plotly.relayout(oel, { shapes: [] });
      });
    });
  });
}

// ── Vehicle selection ─────────────────────────────────────────────────────────
function selectVehicle(reg) {
  selectedVehicle = reg;

  // Highlight matching rows across all tables
  document.querySelectorAll("tr.vehicle-row").forEach((el) => {
    const code = el.querySelector("code");
    el.classList.toggle("selected-row", !!(code && code.textContent.trim() === reg));
  });

  loadSohBands(reg);
  loadBayesCoef(reg);
  loadSessions(reg);
  loadAnomalyBreakdown(reg);

  // Scroll to Bollinger bands section
  document.getElementById("section5").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await Promise.allSettled([
    loadOverview().catch((e) => console.error("loadOverview:", e)),
    loadQuintiles().catch((e) => console.error("loadQuintiles:", e)),
    loadVehicles().catch((e) => console.error("loadVehicles:", e)),
    loadBayesCoef(null).catch((e) => console.error("loadBayesCoef:", e)),
    loadAnomalyTiers().catch((e) => console.error("loadAnomalyTiers:", e)),
    loadAnomalyBreakdown().catch((e) => console.error("loadAnomalyBreakdown:", e)),
  ]);
  document.getElementById("loadingOverlay").style.display = "none";
});
