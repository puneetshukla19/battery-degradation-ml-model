"use strict";

/* ─── state ──────────────────────────────────────────────────────────────────── */
let _overview  = null;
let _vehicles  = null;
let _trend     = null;
let _quintiles = null;
let _tiers     = null;

let _hoverTimeout = null;
let _hideTimeout  = null;

/* ─── helpers ────────────────────────────────────────────────────────────────── */
const fmt    = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
const fmtPct = (v)        => (v == null ? "—" : Number(v).toFixed(2) + "%");

function fmtPeriod(first, last) {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const d1 = new Date(first), d2 = new Date(last);
  const y1 = String(d1.getFullYear()).slice(2);
  const y2 = String(d2.getFullYear()).slice(2);
  return `${months[d1.getMonth()]}'${y1} – ${months[d2.getMonth()]}'${y2}`;
}

function rulToYears(days) {
  if (days == null) return "—";
  return (days / 365.25).toFixed(2) + " years";
}

/* ─── init ───────────────────────────────────────────────────────────────────── */
async function init() {
  try {
    const _j = r => r.ok ? r.json() : Promise.reject(r.status);
    const [ov, veh, trend, quint, tiers, scatter, coef, delta, bdTimeline, dists] = await Promise.all([
      fetch("/api/overview/").then(_j),
      fetch("/api/vehicles/").then(_j),
      fetch("/api/fleet-trend/").then(_j),
      fetch("/api/quintiles/").then(_j),
      fetch("/api/anomaly-tiers/").then(_j),
      fetch("/api/soh-scatter/").then(_j),
      fetch("/api/bayes-coef/").then(_j),
      fetch("/api/soh-delta-trend/").then(_j),
      fetch("/api/breakdown-timeline/").then(_j).catch(() => null),
      fetch("/api/distributions/").then(_j).catch(() => null),
    ]);

    _overview  = ov;
    _vehicles  = veh.vehicles;
    _trend     = trend.trend;
    _quintiles = quint.quintiles;
    _tiers     = tiers;
    _deltaData = delta;

    renderKPICards();
    renderFleetHealth();
    renderSohScatter(scatter);
    renderSohDeltaChart();
    buildVehicleSlider();
    renderQuintiles();
    renderBayesCoef(coef);
    if (bdTimeline && bdTimeline.ref_date) data_refDate = bdTimeline.ref_date;
    renderBreakdownTimeline(bdTimeline);
    renderDistributions(dists);
    renderAnomalyTiers();
    setupHoverCharts();
    // Initialise Bootstrap tooltips (Popper.js-based — not clipped by overflow containers)
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el =>
      new bootstrap.Tooltip(el, { trigger: "hover", placement: "top" })
    );

    // ── Scroll-reveal animations (IntersectionObserver) ──────────────────
    initScrollReveal();

  } catch (e) {
    console.error("executive_summary init failed:", e);
  } finally {
    document.getElementById("loadingOverlay").style.display = "none";
  }
}

/* ─── KPI cards ──────────────────────────────────────────────────────────────── */
function renderKPICards() {
  const o = _overview;
  document.getElementById("kpiPeriod").textContent       = fmtPeriod(o.first_date, o.last_date);
  document.getElementById("kpiVehicles").textContent     = o.n_vehicles;
  // Update hero callout + navbar vehicle count dynamically
  const heroVeh = document.getElementById("heroVehicleCount");
  if (heroVeh && o.n_vehicles != null) heroVeh.textContent = o.n_vehicles;
  const navVeh = document.getElementById("navVehicleCount");
  if (navVeh && o.n_vehicles != null) navVeh.textContent = o.n_vehicles;
  document.getElementById("kpiMeanSoh").textContent      = fmtPct(o.fleet_mean_soh);
  document.getElementById("kpiStdSoh").textContent       = fmtPct(o.fleet_std_soh);
  document.getElementById("kpiEkfRul").textContent       = rulToYears(o.median_ekf_rul);
  document.getElementById("kpiRemainingEfc").textContent = o.fleet_median_remaining_efc != null
    ? Math.round(o.fleet_median_remaining_efc).toLocaleString() + " EFC" : "—";
}

/* ─── Fleet Health table ─────────────────────────────────────────────────────── */
function renderFleetHealth() {
  const o = _overview;
  document.getElementById("ht_span").textContent       = `${o.span_days} days`;
  document.getElementById("ht_meanSoh").textContent    = fmtPct(o.fleet_mean_soh);
  document.getElementById("ht_stdSoh").textContent     = fmtPct(o.fleet_std_soh);
  const sign = o.soh_trend_pct >= 0 ? "+" : "";
  document.getElementById("ht_trend").textContent      = `${sign}${fmt(o.soh_trend_pct, 2)}%`;
  const total = o.total_sessions;
  document.getElementById("ht_population").textContent = total != null
    ? `${total.toLocaleString()} sessions` : "—";
  document.getElementById("ht_ekfRul").textContent     = rulToYears(o.median_ekf_rul);
  if (o.ekf_rul_p25 != null && o.ekf_rul_p75 != null) {
    document.getElementById("ht_rulCi").textContent =
      `${rulToYears(o.ekf_rul_p25)} – ${rulToYears(o.ekf_rul_p75)}`;
  } else {
    document.getElementById("ht_rulCi").textContent = "—";
  }
  document.getElementById("ht_eol").textContent        = `${o.eol_threshold}%`;
}

/* ─── Quintile table ─────────────────────────────────────────────────────────── */
function renderQuintiles() {
  document.querySelector("#quintileTable tbody").innerHTML =
    _quintiles.map(q =>
      `<tr>
        <td>${q.quintile}</td>
        <td class="text-end">${fmt(q.median_soh, 2)}%</td>
      </tr>`
    ).join("");
}

/* ─── BMS SoH vs EKF SoH scatter ────────────────────────────────────────────── */
let _scatterData = null;   // raw points cache for slider re-renders
let _deltaData   = null;   // delta trend cache for slider re-renders
let _scatterVeh  = null;   // currently selected vehicle (null = fleet view)

function renderSohScatter(data) {
  _scatterData = data;
  _drawScatter();
}

function _drawScatter() {
  const el = document.getElementById("sohScatterPlot");
  if (!el || !_scatterData || !_scatterData.points || !_scatterData.points.length) return;
  const pts = _scatterData.points;

  let traces;
  if (_scatterVeh) {
    // Selected vehicle: blue; others: grey
    const selPts   = pts.filter(p => p.registration_number === _scatterVeh);
    const otherPts = pts.filter(p => p.registration_number !== _scatterVeh);
    traces = [
      { type: "scatter", mode: "markers",
        x: otherPts.map(p => p.soh), y: otherPts.map(p => p.ekf_soh),
        marker: { color: "#cbd5e1", size: 3, opacity: 0.35 },
        name: "Other vehicles", hoverinfo: "skip", showlegend: true },
      { type: "scatter", mode: "markers",
        x: selPts.map(p => p.soh), y: selPts.map(p => p.ekf_soh),
        marker: { color: "#3b82f6", size: 5, opacity: 0.8 },
        name: _scatterVeh,
        hovertemplate: _scatterVeh + "<br>BMS: %{x:.2f}%<br>EKF: %{y:.2f}%<extra></extra>",
        showlegend: true },
    ];
  } else {
    traces = [
      { type: "scatter", mode: "markers",
        x: pts.map(p => p.soh), y: pts.map(p => p.ekf_soh),
        marker: { color: "#3b82f6", size: 3.5, opacity: 0.45 },
        hovertemplate: "%{customdata}<br>BMS SoH: %{x:.2f}%<br>EKF SoH: %{y:.2f}%<extra></extra>",
        customdata: pts.map(p => p.registration_number || ""),
        name: "Fleet", showlegend: false },
    ];
  }
  // Compute OLS slope (y on x) from all fleet points
  const allPts  = _scatterData.points || [];
  let slopeAnnotation = null;
  if (allPts.length > 1) {
    const xs = allPts.map(p => p.soh).filter(v => v != null);
    const ys = allPts.map((p, i) => p.soh != null ? p.ekf_soh : null).filter(v => v != null);
    const n  = Math.min(xs.length, ys.length);
    if (n > 1) {
      const xArr = xs.slice(0, n), yArr = ys.slice(0, n);
      const xMu  = xArr.reduce((s, v) => s + v, 0) / n;
      const yMu  = yArr.reduce((s, v) => s + v, 0) / n;
      const cov  = xArr.reduce((s, v, i) => s + (v - xMu) * (yArr[i] - yMu), 0);
      const varX = xArr.reduce((s, v) => s + (v - xMu) ** 2, 0);
      const slope = varX > 0 ? cov / varX : null;
      if (slope !== null) {
        slopeAnnotation = {
          x: 0.98, y: 0.04, xref: "paper", yref: "paper",
          text: `OLS slope: <b>${slope.toFixed(3)}</b>  (EKF = ${slope.toFixed(3)} × BMS + c)`,
          showarrow: false, align: "right",
          font: { size: 9, color: "#475569", family: "Plus Jakarta Sans" },
          bgcolor: "rgba(255,255,255,.88)", borderpad: 4,
          bordercolor: "#e2e8f0", borderwidth: 1,
        };
      }
    }
  }

  // y = x reference line in zoomed range
  traces.push({ type: "scatter", mode: "lines",
    x: [95, 100], y: [95, 100],
    line: { color: "#94a3b8", width: 1.5, dash: "dash" },
    name: "y = x", hoverinfo: "skip" });

  Plotly.react(el, traces, {
    paper_bgcolor: "white", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    margin: { l: 52, r: 12, t: 12, b: 52 },
    xaxis: { title: { text: "BMS-Reported SoH (%)", font: { size: 9.5 } },
             range: [100, 95], gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    yaxis: { title: { text: "EKF SoH (%)", font: { size: 9.5 } },
             range: [100, 95], gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    annotations: slopeAnnotation ? [slopeAnnotation] : [],
    showlegend: _scatterVeh != null,
    legend: { x: 0.02, y: 0.98, font: { size: 8.5 } },
  }, { displayModeBar: false, responsive: true });
}

/* ─── SoH Delta time-series chart (EKF − BMS) ───────────────────────────────── */
function renderSohDeltaChart() {
  _drawDeltaChart();
}

function _drawDeltaChart() {
  const el = document.getElementById("sohDeltaPlot");
  if (!el || !_deltaData) return;

  let traces;
  if (_scatterVeh) {
    // Show only the selected vehicle's per-session delta
    const pts = (_deltaData.vehicle_points || []).filter(p => p.registration_number === _scatterVeh);
    pts.sort((a, b) => a.date < b.date ? -1 : 1);
    traces = [{
      type: "scatter", mode: "lines+markers",
      x: pts.map(p => p.date), y: pts.map(p => p.delta),
      line: { color: "#3b82f6", width: 2 },
      marker: { size: 4, color: "#3b82f6" },
      name: _scatterVeh,
      hovertemplate: "%{x}<br>EKF − BMS delta: %{y:.2f}%<extra></extra>",
    }];
  } else {
    // Fleet-wide daily median delta
    const ft = _deltaData.fleet_trend || [];
    ft.sort((a, b) => a.date < b.date ? -1 : 1);
    traces = [{
      type: "scatter", mode: "lines",
      x: ft.map(p => p.date), y: ft.map(p => p.fleet_median_delta),
      line: { color: "#6366f1", width: 2 },
      fill: "tozeroy", fillcolor: "rgba(99,102,241,0.08)",
      name: "Fleet median δ",
      hovertemplate: "%{x}<br>Fleet median EKF−BMS: %{y:.2f}%<extra></extra>",
    }];
  }
  // Zero reference
  traces.push({ type: "scatter", mode: "lines",
    x: (traces[0].x || []).slice(0, 1).concat((traces[0].x || []).slice(-1)),
    y: [0, 0],
    line: { color: "#94a3b8", width: 1, dash: "dash" },
    name: "Zero line", hoverinfo: "skip", showlegend: false });

  Plotly.react(el, traces, {
    paper_bgcolor: "white", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    margin: { l: 52, r: 12, t: 12, b: 52 },
    xaxis: { title: { text: "Date", font: { size: 9.5 } }, gridcolor: "#e2e8f0", tickangle: -30, tickfont: { size: 8.5 } },
    yaxis: { title: { text: "EKF SoH − BMS SoH (%)", font: { size: 9.5 } }, gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, font: { size: 8.5 } },
  }, { displayModeBar: false, responsive: true });
}

/* ─── Vehicle slider for scatter + delta charts ──────────────────────────────── */
function buildVehicleSlider() {
  const container = document.getElementById("vehSliderContainer");
  if (!container || !_deltaData) return;

  // Get unique sorted vehicle list
  const allVeh = [...new Set((_deltaData.vehicle_points || []).map(p => p.registration_number))].sort();
  if (!allVeh.length) { container.style.display = "none"; return; }

  container.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:8px;padding:8px 14px;margin-bottom:10px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:8px;flex:0 0 auto">
        <label style="font-size:.74rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap">Vehicle</label>
        <input type="range" id="vehSlider" min="0" max="${allVeh.length - 1}" value="0" step="1"
          style="width:180px;accent-color:#3b82f6" ${_scatterVeh ? "" : "disabled"}>
        <span id="vehSliderLabel" style="font-size:.82rem;font-weight:700;color:#3b82f6;min-width:120px">—</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;flex:0 0 auto">
        <input type="checkbox" id="vehSliderToggle" style="accent-color:#3b82f6;width:14px;height:14px">
        <label for="vehSliderToggle" style="font-size:.74rem;color:#64748b;cursor:pointer">Enable vehicle filter</label>
      </div>
      <div style="font-size:.72rem;color:#94a3b8;flex:1;text-align:right">${allVeh.length} vehicles · sorted alphabetically</div>
    </div>
  `;

  const slider = document.getElementById("vehSlider");
  const toggle = document.getElementById("vehSliderToggle");
  const label  = document.getElementById("vehSliderLabel");

  function updateFromSlider() {
    if (!toggle.checked) {
      _scatterVeh = null;
      label.textContent = "Fleet view";
      label.style.color = "#94a3b8";
      slider.disabled = true;
    } else {
      _scatterVeh = allVeh[+slider.value];
      label.textContent = _scatterVeh;
      label.style.color = "#3b82f6";
      slider.disabled = false;
    }
    _drawScatter();
    _drawDeltaChart();
  }

  toggle.addEventListener("change", updateFromSlider);
  slider.addEventListener("input",  updateFromSlider);
  updateFromSlider();  // initial state = fleet view
}

/* ─── Key Degradation Signals — plain-English definitions for ⓘ hover ─────────── */
const SIGNAL_DEFS = {
  days_since_first_session: "Total calendar days since this vehicle's first recorded session — captures time-based degradation independent of usage.",
  days_since_first:         "Total calendar days since this vehicle's first recorded session — captures time-based degradation independent of usage.",
  cum_efc:                  "Cumulative equivalent full charge cycles — counts the total charge throughput normalised to one full 0→100% charge.",
  thermal_stress:           "Composite measure of heat stress: integrates temperature rise rate and high-temperature exposure across sessions.",
  dod_stress:               "Depth-of-discharge stress — deeper discharges accelerate cathode degradation; higher values mean more aggressive cycling.",
  ir_ohm_mean_ewm10:        "Exponentially-weighted moving average of cell internal resistance (Ω). Rising IR signals electrolyte ageing or contact loss.",
  ir_ohm_trend_slope:       "Long-run slope of internal resistance over time. A positive slope means IR is systematically growing.",
  ir_event_trend_slope:     "Trend in the rate of high-IR events (spikes above threshold). Increasing events indicate accelerating degradation.",
  ir_event_rate:            "Fraction of sessions with at least one high-IR event — a proxy for how often the pack is operating outside healthy limits.",
  cell_spread_mean_ewm10:   "EWM-smoothed average of max−min cell voltage spread. Larger spread means cells are diverging in capacity or health.",
  spread_trend_slope:       "Trend in cell voltage spread over time. A rising slope means cells are diverging faster.",
  vsag_rate_per_hr_ewm10:   "Rate of voltage sag events per hour of operation (EWM-smoothed). Frequent sags indicate rising polarisation or weak cells.",
  vsag_trend_slope:         "Trend in the voltage sag rate over time. A positive slope means sag events are becoming more frequent.",
  n_vsag:                   "Count of voltage sag events in the session — momentary dips under load pointing to high impedance or capacity fade.",
  temp_rise_rate_ewm10:     "EWM-smoothed rate of temperature rise during operation (°C/min). Faster heating indicates increasing internal resistance.",
  energy_per_km:            "Energy consumed per kilometre of travel (kWh/km). Higher values may reflect pack inefficiency or heavier loads.",
  energy_per_loaded_session:"Energy delivered per session, normalised for load type (loaded vs unloaded). Captures efficiency trends across trip types.",
  energy_kwh:               "Total energy throughput in the session (kWh). High throughput sessions contribute more to electrochemical wear.",
  capacity_ah_discharge_new:"Raw motoring Ah delivered this session (hves1 sensor). Large values represent high-throughput discharge stress.",
  block_capacity_ah:        "Cumulative Ah discharged across the full discharge block. Reflects the total electrochemical work in one block.",
  c_rate_chg:               "Charge C-rate — ratio of charging current to rated capacity. Higher C-rates cause more lithium plating and heat stress.",
  charging_rate_kw:         "Average charging power (kW). Faster charging increases heat and lithium plating risk.",
  cycle_soh:                "Per-cycle SoH estimate (capacity method). Direct measure of capacity fade from one full discharge cycle.",
  aging_index:              "Composite aging index combining calendar and cycle stress into a single degradation predictor.",
  soh_trend_slope:          "Local slope of BMS-reported SoH over recent sessions — the rate at which the BMS sees capacity declining.",
  n_high_ir:                "Count of sessions with abnormally high internal resistance events — a proxy for electrolyte or SEI layer degradation.",
  n_low_soc:                "Count of sessions where SoC dipped to a critically low level. Deep discharges accelerate cathode stress.",
  odometer_km:              "This session's trip distance (km). Longer trips mean more discharge depth per session.",
  block_odometer_km:        "Cumulative trip distance across the whole discharge block (km).",
  duration_hr:              "Session duration in hours. Longer sessions accumulate more electrochemical stress.",
  is_loaded:                "Whether the vehicle carried a load. Loaded trips draw more current and discharge deeper.",
  speed_mean:               "Mean travel speed (km/h). Higher speeds typically correlate with higher motor current and discharge rate.",
};

/* ─── Key Degradation Signals (Bayesian ridge coefficients — negative only) ───── */
function renderBayesCoef(data) {
  const el = document.getElementById("bayesCoefPlot");
  if (!el || !data || !data.global) return;

  const raw = data.global;
  // Keep only negative coefficients (signals that drive SoH decline), sort most harmful first
  const sorted = Object.entries(raw)
    .filter(([, v]) => v != null && isFinite(v) && v < 0)
    .sort((a, b) => a[1] - b[1])   // most negative first
    .slice(0, 14);

  if (!sorted.length) return;

  const labels   = sorted.map(([k]) => (COEF_LABEL_MAP[k] || k) + " ⓘ");
  const values   = sorted.map(([, v]) => +Math.abs(v).toFixed(5));
  const defs     = sorted.map(([k]) => SIGNAL_DEFS[k] || "No definition available.");
  const rawKeys  = sorted.map(([k]) => k);

  Plotly.newPlot(el, [{
    type: "bar",
    x: labels,
    y: values,
    customdata: defs,
    marker: { color: "#ef4444", opacity: 0.8 },
    hovertemplate: "<b>%{x}</b><br>Degradation weight: <b>%{y:.5f}</b><br><br><span style='color:white;font-style:italic'>%{customdata}</span><extra></extra>",
  }], {
    paper_bgcolor: "white", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    margin: { l: 44, r: 12, t: 12, b: 130 },
    xaxis: { tickfont: { size: 10.5 }, tickangle: -38, automargin: true },
    yaxis: { title: { text: "Degradation weight", font: { size: 9 } },
             gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    showlegend: false,
  }, { displayModeBar: false, responsive: true });

  // ── Populate description paragraph below the chart ──────────────────────
  const descEl = document.getElementById("bayesCoefDesc");
  if (descEl && sorted.length) {
    // Build natural-language sentences for top signals
    const unitOf = (k) => {
      if (k.includes("day") || k === "days_since_first_session") return "calendar day";
      if (k === "cum_efc")             return "equivalent full cycle";
      if (k === "thermal_stress")      return "unit of thermal stress";
      if (k === "dod_stress")          return "unit of DoD stress";
      if (k.includes("ir_ohm"))        return "mΩ of internal resistance";
      if (k.includes("ir_event"))      return "IR event";
      if (k.includes("spread"))        return "mV of cell spread";
      if (k.includes("vsag"))          return "V-sag event";
      if (k.includes("temp"))          return "°C/min of temp rise";
      if (k.includes("energy_per_km")) return "kWh/km";
      if (k.includes("energy_kwh") || k.includes("energy_per_loaded")) return "kWh";
      if (k.includes("capacity_ah"))   return "Ah";
      if (k.includes("c_rate") || k.includes("charging_rate")) return "C-rate unit";
      if (k.includes("cycle_soh") || k.includes("soh_trend")) return "% SoH unit";
      if (k.includes("odometer") || k === "odometer_km") return "km";
      if (k.includes("duration"))      return "hour";
      return "unit";
    };

    const top = sorted.slice(0, 5);
    const sentences = top.map(([k, v]) => {
      const label = COEF_LABEL_MAP[k] || k;
      const weight = Math.abs(v).toFixed(4);
      return `<b>${label}</b> (−${weight}% per ${unitOf(k)})`;
    });

    const intro = `The Bayesian Ridge model identifies the following as the strongest drivers of SoH decline across the fleet: `;
    const body  = sentences.join(", ") + ".";
    const outro = sorted.length > 5
      ? ` ${sorted.length - 5} additional signal${sorted.length - 5 > 1 ? "s" : ""} contribute smaller but non-zero degradation weight. Hover any bar for a plain-English definition.`
      : ` Hover any bar for a plain-English definition.`;

    descEl.innerHTML = intro + body + outro;
  }
}

/* ─── Anomaly tiers ──────────────────────────────────────────────────────────── */
function renderAnomalyTiers() {
  const d = _tiers;

  const eol = (_overview && _overview.eol_threshold) || 80;
  // Compute RUL the same way as the RUL scatter chart: (SoH headroom) / |daily slope|
  const rulDisplay = (v) => {
    const slope = v.soh_slope;
    if (v.current_soh == null || slope == null || slope >= 0) return "—";
    const days = Math.max(0, Math.round((v.current_soh - eol) / Math.abs(slope)));
    return `${days.toLocaleString()} d<br><span style="color:#94a3b8;font-size:.75rem">(${(days/365.25).toFixed(1)} yr)</span>`;
  };

  const vRow = (v, signal, color) =>
    `<tr class="tier-vehicle-row" data-reg="${v.registration_number}"
         style="cursor:pointer;transition:background .12s"
         onclick="openVehicleDetail('${v.registration_number}');_tierMarkActive(this)">
      <td>
        <a class="tier-reg-link" style="font-size:.8rem;font-weight:700;color:${color};
           text-decoration:underline;text-underline-offset:2px;cursor:pointer;
           display:inline-flex;align-items:center;gap:4px"
           title="Click to open vehicle detail">
          ${v.registration_number}
          <span style="font-size:.65rem;opacity:.75">↗</span>
        </a>
      </td>
      <td class="text-muted">${signal || ""}</td>
      <td class="text-end">${fmtPct(v.current_soh)}</td>
      <td class="text-end">${fmt(v.soh_slope, 4)}</td>
      <td class="text-end fw-bold">${v.composite != null ? (v.composite * 100).toFixed(1) : "—"}</td>
      <td class="text-end">${rulDisplay(v)}</td>
      <td class="text-end">${v.n_combined_anom}</td>
    </tr>`;

  document.querySelector("#tier1Table tbody").innerHTML = d.tier1.map(v => vRow(v, v.primary_signal, "#dc2626")).join("");
  document.querySelector("#tier2Table tbody").innerHTML = d.tier2.map(v => vRow(v, v.note,           "#d97706")).join("");
  document.querySelector("#tier3Table tbody").innerHTML = d.tier3.map(v => vRow(v, v.note,           "#059669")).join("");
}

/* ─── Tier row active state ──────────────────────────────────────────────────── */
function _tierMarkActive(row) {
  document.querySelectorAll(".tier-vehicle-row.tier-active").forEach(r => r.classList.remove("tier-active"));
  row.classList.add("tier-active");
}

/* ─── Hover chart setup ──────────────────────────────────────────────────────── */
const HOVER_FNS = {
  // Fleet health table row hovers
  mean_soh:          chartVehicleSoh,
  std_soh:           chartSohStdDev,
  trend:             chartSohTrend,
  ekf_rul:           chartVehicleRul,
  eol:               chartEolInfo,
  data_span:         chartDataSpan,
  population:        chartPopulation,
  // KPI card hovers
  kpi_mean_soh:      chartSohHistogram,
  kpi_std_soh:       chartSohStdDev,
  kpi_rul:           chartRulHistogram,
  kpi_remaining_efc: chartRemainingEfc,
  kpi_period:        chartDataSpan,
};

function setupHoverCharts() {
  const panel = document.getElementById("hoverPanel");

  // Fleet health table row hovers
  document.querySelectorAll("[data-hover]").forEach(row => {
    row.addEventListener("mouseenter", () => {
      clearTimeout(_hideTimeout);
      clearTimeout(_hoverTimeout);
      _hoverTimeout = setTimeout(() => showHoverChart(row.dataset.hover, row), 130);
    });
    row.addEventListener("mouseleave", () => {
      clearTimeout(_hoverTimeout);
      _hideTimeout = setTimeout(hideHoverChart, 220);
    });
  });

  // KPI card hovers
  document.querySelectorAll("[data-kpi-hover]").forEach(card => {
    card.addEventListener("mouseenter", () => {
      clearTimeout(_hideTimeout);
      clearTimeout(_hoverTimeout);
      _hoverTimeout = setTimeout(() => showHoverChart(card.dataset.kpiHover, card), 200);
    });
    card.addEventListener("mouseleave", () => {
      clearTimeout(_hoverTimeout);
      _hideTimeout = setTimeout(hideHoverChart, 300);
    });
  });

  // (i) icons inside KPI cards — tooltip takes priority; suppress the chart hover
  document.querySelectorAll("[data-kpi-hover] .col-info").forEach(icon => {
    icon.addEventListener("mouseenter", e => {
      e.stopPropagation();          // prevent card mouseenter from firing
      clearTimeout(_hoverTimeout);  // cancel any in-flight chart delay
      hideHoverChart();             // dismiss chart if already showing
    });
    icon.addEventListener("mouseleave", e => {
      e.stopPropagation();          // prevent card mouseleave from scheduling hide
    });
  });

  panel.addEventListener("mouseenter", () => clearTimeout(_hideTimeout));
  panel.addEventListener("mouseleave", () => {
    _hideTimeout = setTimeout(hideHoverChart, 220);
  });
}

function showHoverChart(type, rowEl) {
  const panel  = document.getElementById("hoverPanel");
  const plotEl = document.getElementById("hoverPlot");
  const textEl = document.getElementById("hoverText");
  const rect   = rowEl.getBoundingClientRect();
  const isText = type === "eol";
  const panelH = isText ? 180 : 300;

  // Position panel
  const panelW = 480;
  let left = rect.right + 14;
  if (left + panelW > window.innerWidth) left = rect.left - panelW - 14;
  left = Math.max(8, left);

  let top = rect.top - 4;
  if (top + panelH > window.innerHeight - 8) top = Math.max(8, window.innerHeight - panelH - 8);
  top = Math.max(8, top);

  panel.style.left    = left + "px";
  panel.style.top     = top + "px";
  panel.style.height  = isText ? "auto" : panelH + "px";
  panel.style.display = "block";

  plotEl.style.display = isText ? "none" : "block";
  textEl.style.display = isText ? "block" : "none";

  // Purge any previous Plotly graph so layout dimensions are applied fresh,
  // then render after the browser has committed the display:block paint.
  if (!isText) Plotly.purge(plotEl);

  requestAnimationFrame(() => requestAnimationFrame(() => {
    const fn = HOVER_FNS[type];
    if (fn) fn(plotEl, textEl);
  }));
}

function hideHoverChart() {
  document.getElementById("hoverPanel").style.display = "none";
}

/* ─── Chart: vehicle SoH bar ─────────────────────────────────────────────────── */
function chartVehicleSoh(plotEl) {
  // Sort descending so highest SoH is leftmost
  const sorted = [..._vehicles]
    .filter(v => v.current_soh != null)
    .sort((a, b) => b.current_soh - a.current_soh);

  const colors = sorted.map(v =>
    v.current_soh >= 97 ? "#22c55e" :
    v.current_soh >= 95 ? "#f59e0b" : "#ef4444"
  );

  Plotly.newPlot(plotEl, [{
    type: "bar",
    x: sorted.map(v => v.registration_number),
    y: sorted.map(v => v.current_soh),
    marker: { color: colors },
    hovertemplate: "%{x}<br>SoH: %{y:.2f}%<extra></extra>",
  }], {
    ...baseLayout("EKF SoH per vehicle"),
    width: undefined,
    height: 282,
    xaxis: { tickfont: { size: 7.5, color: "#1e293b" }, tickangle: -40, automargin: true },
    yaxis: { ...yAx("SoH (%)"), range: [93, 100] },
    margin: { t: 34, b: 70, l: 46, r: 14 },
  }, { displayModeBar: false, responsive: true });
}

/* ─── Chart: SoH std dev histogram ──────────────────────────────────────────── */
function chartSohStdDev(plotEl) {
  const sohVals = _vehicles.map(v => v.current_soh).filter(v => v != null);
  const mu  = _overview.fleet_mean_soh;
  const sig = _overview.fleet_std_soh;
  const lo  = mu - 1.96 * sig;
  const hi  = Math.min(mu + 1.96 * sig, 100);  // cap at 100%

  Plotly.newPlot(plotEl, [{
    type: "histogram",
    x: sohVals,
    nbinsx: 6,
    marker: { color: "#3b82f6", opacity: 0.75 },
    hovertemplate: "SoH: %{x:.2f}%<br>Count: %{y}<extra></extra>",
  }], {
    ...baseLayout(`SoH Distribution   µ=${mu.toFixed(3)}%  σ=${sig.toFixed(3)}%`),
    xaxis: { ...xAx(), title: { text: "SoH (%)", font: { size: 9 } } },
    yaxis: { ...yAx("Count") },
    shapes: [
      {
        // 95% confidence region (±1.96σ)
        type: "rect",
        x0: lo, x1: hi, y0: 0, y1: 1,
        xref: "x", yref: "paper",
        fillcolor: "rgba(59,130,246,0.10)",
        line: { width: 0 },
      },
      {
        // Mean line
        type: "line",
        x0: mu, x1: mu, y0: 0, y1: 1,
        xref: "x", yref: "paper",
        line: { color: "#1e293b", width: 1.5, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: mu, y: 0.98, xref: "x", yref: "paper",
        text: `µ=${mu.toFixed(3)}%`, showarrow: false,
        font: { size: 8.5, color: "#1e293b", family: "Plus Jakarta Sans" },
        bgcolor: "rgba(255,255,255,0.8)", borderpad: 2,
      },
      {
        x: lo, y: 0.5, xref: "x", yref: "paper",
        text: "−1.96σ", showarrow: false,
        font: { size: 8, color: "#64748b", family: "Plus Jakarta Sans" },
      },
      {
        x: hi, y: 0.5, xref: "x", yref: "paper",
        text: "+1.96σ", showarrow: false,
        font: { size: 8, color: "#64748b", family: "Plus Jakarta Sans" },
      },
    ],
  }, cfg());
}

/* ─── rolling median helper (removes single-day composition spikes) ──────────── */
function rollingMedian(arr, win) {
  const half = Math.floor(win / 2);
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i - half), Math.min(arr.length, i + half + 1))
                     .slice().sort((a, b) => a - b);
    const mid = Math.floor(slice.length / 2);
    return slice.length % 2 !== 0 ? slice[mid] : (slice[mid - 1] + slice[mid]) / 2;
  });
}

/* ─── Chart: SoH trend line ──────────────────────────────────────────────────── */
function chartSohTrend(plotEl) {
  if (!_trend || !_trend.length) return;

  const dates  = _trend.map(r => r.date);
  const sohs   = _trend.map(r => r.median_soh);
  const first  = sohs[0];
  const last   = sohs[sohs.length - 1];
  const slope  = _overview.soh_trend_pct;
  const sign   = slope >= 0 ? "+" : "";
  const lineColor = last < first ? "#ef4444" : "#3b82f6";
  const fillColor = last < first ? "rgba(239,68,68,0.08)" : "rgba(59,130,246,0.08)";

  const minY = Math.min(...sohs);
  const maxY = Math.max(...sohs);
  const pad  = Math.max((maxY - minY) * 0.3, 0.05);

  Plotly.newPlot(plotEl, [
    {
      type: "scatter", mode: "lines",
      x: dates, y: sohs,
      line: { color: lineColor, width: 2, shape: "spline", smoothing: 0.8 },
      fill: "tonexty", fillcolor: fillColor,
      hovertemplate: "%{x}<br>Fleet median SoH: %{y:.3f}%<extra></extra>",
      name: "Fleet median SoH",
    },
  ], {
    ...baseLayout(`Fleet SoH Trend   ${sign}${slope.toFixed(2)}% over ${_overview.span_days} days`),
    yaxis: { ...yAx("EKF SoH (%)"), range: [minY - pad, maxY + pad] },
    xaxis: { ...xAx() },
  }, cfg());
}

/* ─── Chart: vehicle RUL bar ─────────────────────────────────────────────────── */
function chartVehicleRul(plotEl) {
  const field = v => v.ekf_rul_days != null ? v.ekf_rul_days : v.rul_days;
  const all = [..._vehicles].filter(v => field(v) != null);

  // Outlier removal: exclude values above Q3 + 3×IQR
  const vals  = all.map(v => field(v)).sort((a, b) => a - b);
  const q1    = vals[Math.floor(vals.length * 0.25)];
  const q3    = vals[Math.floor(vals.length * 0.75)];
  const fence = q3 + 3 * (q3 - q1);

  // Sort descending so longest RUL is leftmost
  const sorted = all
    .filter(v => field(v) <= fence)
    .sort((a, b) => field(b) - field(a));

  const colors = sorted.map(v =>
    field(v) > 730 ? "#22c55e" :
    field(v) > 365 ? "#f59e0b" : "#ef4444"
  );

  Plotly.newPlot(plotEl, [{
    type: "bar",
    x: sorted.map(v => v.registration_number),
    y: sorted.map(v => +(field(v) / 365.25).toFixed(2)),
    marker: { color: colors },
    hovertemplate: "%{x}<br>RUL: %{y:.2f} yr<extra></extra>",
  }], {
    ...baseLayout("EKF RUL per vehicle"),
    width: undefined,
    height: 282,
    xaxis: { tickfont: { size: 7.5, color: "#1e293b" }, tickangle: -40, automargin: true },
    yaxis: { ...yAx("RUL (years)") },
    margin: { t: 34, b: 70, l: 46, r: 14 },
  }, { displayModeBar: false, responsive: true });
}

/* ─── Chart: SoH distribution histogram (KPI card hover) ────────────────────── */
function chartSohHistogram(plotEl) {
  const sohVals = _vehicles.map(v => v.current_soh).filter(v => v != null);
  const mu  = _overview.fleet_mean_soh;
  const sig = _overview.fleet_std_soh;
  const lo  = mu - 1.96 * sig;
  const hi  = Math.min(mu + 1.96 * sig, 100);  // cap at 100%

  Plotly.newPlot(plotEl, [{
    type: "histogram",
    x: sohVals,
    nbinsx: 6,
    marker: { color: "#3b82f6", opacity: 0.75 },
    hovertemplate: "SoH: %{x:.2f}%<br>Count: %{y}<extra></extra>",
  }], {
    ...baseLayout(`SoH Distribution   µ=${mu.toFixed(3)}%  σ=${sig.toFixed(3)}%`),
    xaxis: { ...xAx(), title: { text: "SoH (%)", font: { size: 9 } } },
    yaxis: { ...yAx("Count") },
    shapes: [
      { type: "rect", x0: lo, x1: hi, y0: 0, y1: 1,
        xref: "x", yref: "paper", fillcolor: "rgba(59,130,246,0.10)", line: { width: 0 } },
      { type: "line", x0: mu, x1: mu, y0: 0, y1: 1,
        xref: "x", yref: "paper", line: { color: "#1e293b", width: 1.5, dash: "dash" } },
    ],
    annotations: [
      { x: mu, y: 0.98, xref: "x", yref: "paper", text: `µ=${mu.toFixed(3)}%`,
        showarrow: false, font: { size: 8.5, color: "#1e293b", family: "Plus Jakarta Sans" },
        bgcolor: "rgba(255,255,255,0.8)", borderpad: 2 },
      { x: lo, y: 0.5, xref: "x", yref: "paper", text: "−1.96σ",
        showarrow: false, font: { size: 8, color: "#64748b", family: "Plus Jakarta Sans" } },
      { x: hi, y: 0.5, xref: "x", yref: "paper", text: "+1.96σ",
        showarrow: false, font: { size: 8, color: "#64748b", family: "Plus Jakarta Sans" } },
    ],
  }, cfg());
}

/* ─── Chart: RUL histogram (KPI card hover) ──────────────────────────────────── */
function chartRulHistogram(plotEl) {
  const rulYears = _vehicles
    .filter(v => v.ekf_rul_days != null && v.ekf_rul_days > 0)
    .map(v => +(v.ekf_rul_days / 365.25).toFixed(2));

  if (!rulYears.length) { Plotly.purge(plotEl); return; }

  const med = _overview.median_ekf_rul != null
    ? +(_overview.median_ekf_rul / 365.25).toFixed(2) : null;

  const shapes = [], annotations = [];
  if (med != null) {
    shapes.push({ type: "line", x0: med, x1: med, y0: 0, y1: 1,
      xref: "x", yref: "paper", line: { color: "#1e293b", width: 1.5, dash: "dash" } });
    annotations.push({ x: med, y: 0.98, xref: "x", yref: "paper",
      text: `Median ${med.toFixed(2)} yr`, showarrow: false,
      font: { size: 8.5, color: "#1e293b", family: "Plus Jakarta Sans" },
      bgcolor: "rgba(255,255,255,0.8)", borderpad: 2 });
  }

  const minV = Math.floor(Math.min(...rulYears) * 4) / 4;   // round down to nearest 0.25
  const maxV = Math.ceil(Math.max(...rulYears)  * 4) / 4;   // round up   to nearest 0.25

  Plotly.newPlot(plotEl, [{
    type: "histogram",
    x: rulYears,
    autobinx: false,
    xbins: { start: minV, end: maxV, size: 0.25 },           // 3-month bins
    marker: { color: "#8b5cf6", opacity: 0.75 },
    hovertemplate: "RUL: %{x:.2f}–%{x:.2f} yr<br>Count: %{y}<extra></extra>",
  }], {
    ...baseLayout("EKF RUL Distribution (years)"),
    xaxis: { ...xAx(), title: { text: "RUL (years)", font: { size: 9 } }, dtick: 0.5 },
    yaxis: { ...yAx("Count") },
    shapes,
    annotations,
  }, cfg());
}

/* ─── EoL info text ──────────────────────────────────────────────────────────── */
function chartEolInfo(plotEl, textEl) {
  const mu       = _overview.fleet_mean_soh;
  const eol      = _overview.eol_threshold;
  const headroom = (((mu - eol) / mu) * 100).toFixed(1);

  textEl.innerHTML = `
    <div style="padding:16px 18px">
      <div style="font-size:.85rem;font-weight:700;color:#0f172a;margin-bottom:10px">
        End-of-Life SoH Threshold — ${eol}%
      </div>
      <p style="font-size:.8rem;color:#475569;margin:0 0 10px">
        A battery pack is classified as <strong>end-of-life (EoL)</strong> when its State of Health
        drops below <strong>${eol}%</strong> of its rated capacity.
      </p>
      <p style="font-size:.8rem;color:#475569;margin:0 0 12px">
        Below ${eol}%, capacity fade becomes non-linear and range unpredictability increases
        significantly — making the pack unsuitable for commercial EV operation.
      </p>
      <div style="background:#f0fdf4;border-left:3px solid #16a34a;padding:8px 12px;
                  border-radius:0 4px 4px 0;font-size:.78rem;color:#166534">
        Fleet current mean SoH: <strong>${fmtPct(mu)}</strong> —
        <strong>${headroom}%</strong> of remaining useful capacity before EoL
      </div>
    </div>
  `;
}

async function chartDataSpan(plotEl, textEl) {
  const d = await fetch("/api/fleet-trend/").then(r => r.json());
  const trend = d.trend || [];
  if (!trend.length) { textEl.innerHTML = `<div style="padding:16px;color:#94a3b8">No data.</div>`; return; }

  const dates  = trend.map(r => r.date);
  const pcts   = trend.map(r => r.pct ?? 0);
  const counts = trend.map(r => r.vehicle_count ?? 0);
  const total  = d.total_vehicles ?? 1;

  const h = 360;
  document.getElementById("hoverPanel").style.height = h + "px";
  plotEl.style.height = h + "px";

  Plotly.newPlot(plotEl, [{
    type: "bar", x: dates, y: pcts,
    marker: { color: pcts.map(p => p >= 80 ? "#22c55e" : p >= 40 ? "#f59e0b" : "#ef4444") },
    hovertemplate: "%{x}<br>%{customdata} / " + total + " vehicles (%{y:.1f}%)<extra></extra>",
    customdata: counts,
  }], {
    paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 9.5, color: "#475569" },
    margin: { l: 44, r: 10, t: 10, b: 70 },
    height: h,
    xaxis: { gridcolor: "#e2e8f0", tickangle: -40, tickfont: { size: 8.5 } },
    yaxis: { gridcolor: "#e2e8f0", range: [0, 105], title: { text: "% fleet", font: { size: 9 } } },
    showlegend: false,
  }, { displayModeBar: false, responsive: false });
}

/* ─── Plotly layout helpers ──────────────────────────────────────────────────── */
const FONT = { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" };

function baseLayout(title) {
  return {
    title: {
      text: title,
      font: { ...FONT, size: 10.5, color: "#0f172a" },
      x: 0.02, xanchor: "left",
    },
    width: 448,
    height: 282,
    margin: { t: 34, b: 44, l: 46, r: 14 },
    paper_bgcolor: "white",
    plot_bgcolor: "#f8fafc",
    font: FONT,
    showlegend: false,
  };
}

function yAx(label) {
  return {
    title: { text: label, font: { size: 9 } },
    gridcolor: "#e2e8f0",
    tickfont: FONT,
  };
}

function xAx() {
  return { gridcolor: "#e2e8f0", tickfont: { ...FONT, size: 8.5 } };
}

function cfg() {
  return { displayModeBar: false, responsive: false };
}

/* ─── Chart: Remaining EFC histogram ────────────────────────────────────────── */
function chartRemainingEfc(plotEl) {
  const vals = (_overview.remaining_efc_per_veh || []).filter(v => v != null && v > 0);
  if (!vals.length) {
    plotEl.innerHTML = `<div style="padding:20px;color:#94a3b8;font-size:.82rem;text-align:center">No remaining EFC data available.</div>`;
    return;
  }
  const med = _overview.fleet_median_remaining_efc;
  Plotly.newPlot(plotEl, [{
    type: "histogram",
    x: vals,
    nbinsx: 6,
    marker: { color: "#10b981", opacity: 0.75 },
    hovertemplate: "Remaining EFC: %{x:.0f}<br>Vehicles: %{y}<extra></extra>",
  }], {
    ...baseLayout(`Remaining EFC — fleet  median=${med != null ? Math.round(med) : "—"}`),
    xaxis: { ...xAx(), title: { text: "Estimated Remaining EFC", font: { size: 9 } } },
    yaxis: { ...yAx("Vehicles") },
    shapes: med != null ? [{
      type: "line", x0: med, x1: med, y0: 0, y1: 1,
      xref: "x", yref: "paper",
      line: { color: "#1e293b", width: 1.5, dash: "dash" },
    }] : [],
    annotations: med != null ? [{
      x: med, y: 0.96, xref: "x", yref: "paper",
      text: `median=${Math.round(med)}`, showarrow: false,
      font: { size: 8.5, color: "#1e293b", family: "Plus Jakarta Sans" },
      bgcolor: "rgba(255,255,255,0.8)", borderpad: 2,
    }] : [],
  }, cfg());
}

/* ─── Chart: Population breakdown donut ──────────────────────────────────────── */
function chartPopulation(plotEl) {
  const o        = _overview;
  const charging  = o.charging_sessions  ?? 0;
  const discharge = o.discharge_sessions ?? 0;
  const other     = Math.max(0, (o.total_sessions ?? 0) - charging - discharge);
  const labels = ["Charging", "Discharging", ...(other > 0 ? ["Other"] : [])];
  const values = [charging,  discharge,      ...(other > 0 ? [other]  : [])];

  Plotly.newPlot(plotEl, [{
    type: "pie", hole: 0.52,
    labels, values,
    marker: { colors: ["#f59e0b", "#6366f1", "#94a3b8"] },
    textinfo: "label+percent", textposition: "outside",
    hovertemplate: "%{label}: %{value:,} sessions (%{percent})<extra></extra>",
  }], {
    ...baseLayout(`${(o.total_sessions || 0).toLocaleString()} Total Sessions`),
    height: 260,
    showlegend: false,
    margin: { t: 34, b: 10, l: 10, r: 10 },
  }, cfg());
}

/* ─── Bad anomaly filter (only highlight genuinely harmful sessions) ─────────── */
function isBadAnomaly(s) {
  // CUSUM alarms always track sustained degradation (inherently bad)
  if (s.cusum_ekf_soh_alarm || s.cusum_soh_alarm || s.cusum_cycle_soh_alarm ||
      s.cusum_heat_alarm     || s.cusum_spread_alarm || s.cusum_spread_slope_alarm ||
      s.cusum_epk_alarm      || s.cusum_ir_slope_alarm) return true;
  // IF anomaly: only bad if reason contains known degradation indicators
  if (s.if_anomaly) {
    const reason = (s.if_reason || "").toLowerCase();
    const BAD = ["n_high_ir","ir_ohm","ir_event","n_vsag","d_vsag","cell_spread",
                 "subsystem_voltage","temp","thermal","dod_stress","n_low_soc",
                 "voltage_min","soh","ekf_soh","capacity_soh","cycle_soh",
                 "energy_per_loaded","capacity_ah"];
    if (BAD.some(k => reason.includes(k))) return true;
  }
  return false;
}

/* ─── boot ───────────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  init();
  // Close overlay when clicking the dark backdrop (outside the panel)
  document.getElementById("vdBackdrop").addEventListener("click", function(e) {
    if (e.target === this) closeVehicleDetail();
  });
});

/* ═══════════════════════════════════════════════════════════════════════════════
   VEHICLE DETAIL OVERLAY
   ═══════════════════════════════════════════════════════════════════════════════ */

/* ─── Coef label formatter ───────────────────────────────────────────────────── */
const COEF_LABEL_MAP = {
  dod_stress: "DoD stress", cum_efc: "Cumulative EFC", aging_index: "Aging index",
  days_since_first: "Calendar aging (days)", days_since_first_session: "Calendar Aging",
  soh_trend_slope: "SoH trend slope",
  cycle_soh: "Cycle SoH", ir_ohm_mean_ewm10: "IR mean EWM10",
  cell_spread_mean_ewm10: "Cell spread EWM10", vsag_rate_per_hr_ewm10: "V-sag rate EWM10",
  temp_rise_rate_ewm10: "Temp rise EWM10", ir_event_trend_slope: "IR event trend slope",
  ir_ohm_trend_slope: "IR trend slope", spread_trend_slope: "Spread trend slope",
  vsag_trend_slope: "V-sag trend slope", ir_event_rate: "IR event rate",
  energy_per_km: "Energy per km", energy_kwh: "Energy kWh",
  energy_per_loaded_session: "Energy per loaded session",
  capacity_ah_discharge_new: "Cap AH discharged (hves1)", capacity_ah_charge_new: "Cap AH regen (hves1)",
  capacity_ah_plugin_new: "Cap AH plugin (hves1)", capacity_ah_charge_total_new: "Cap AH chrg. total (hves1)",
  block_capacity_ah: "Block capacity AH", block_odometer_km: "Block odometer km",
  charging_rate_kw: "Charge rate kW", thermal_stress: "Thermal stress",
  c_rate_chg: "Charge C-rate", is_loaded: "Is loaded", odometer_km: "Odometer km",
  duration_hr: "Duration hr", n_vsag: "V-sag count", n_high_ir: "High IR count",
  n_low_soc: "Low SoC count", bms_coverage: "BMS coverage",
  weak_subsystem_consistency: "Weak subsystem consistency",
  hot_subsystem_consistency: "Hot subsystem consistency",
  subsystem_voltage_std: "Subsystem voltage STD", total_alerts: "Total alerts",
  cell_health_poor: "Cell health poor", n_cell_undervoltage: "Cell undervoltage count",
  n_cell_overvoltage: "Cell overvoltage count", rapid_heating: "Rapid heating",
  high_energy_per_km: "High energy per km", slow_charging: "Slow Charge rate",
  fast_charging: "Fast Charge rate", ref_capacity_ah: "Reference capacity AH",
  cell_spread_max: "Cell spread max", speed_mean: "Speed mean",
};

function formatCoefLabel(name) {
  if (COEF_LABEL_MAP[name]) return COEF_LABEL_MAP[name];
  // Fallback: sentence case, preserve known acronyms
  const acro = { ir: "IR", soh: "SoH", ekf: "EKF", dod: "DoD", efc: "EFC",
                 ewm10: "EWM10", ewm: "EWM", std: "STD", bms: "BMS", ah: "AH",
                 kw: "kW", km: "km", chg: "Charge" };
  return name.split("_").map((w, i) => {
    const l = w.toLowerCase();
    if (acro[l]) return acro[l];
    return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase() : w.toLowerCase();
  }).join(" ");
}

/* ─── RUL explanation generator ──────────────────────────────────────────────── */
function generateRulAnalysis(vehicle) {
  const soh    = vehicle.current_soh;
  const slope  = vehicle["soh_slope_%per_day"];
  const eol    = (_overview && _overview.eol_threshold) || 80;
  const rel    = vehicle.rul_reliability;

  const headroom = soh != null ? +(soh - eol).toFixed(2) : null;
  // Single canonical RUL: (SoH headroom) / |daily slope| — same formula used in scatter chart
  const rulDays  = (slope && slope < 0 && headroom != null)
                   ? Math.max(0, Math.round(headroom / Math.abs(slope))) : null;

  const lines = [];

  if (rulDays != null) {
    lines.push(`Estimated RUL: <strong>${rulDays.toLocaleString()} days</strong> ` +
      `(${(rulDays/365.25).toFixed(1)} yr). ` +
      `With ${headroom}% of SoH headroom remaining to the ${eol}% EoL threshold and a slope of ` +
      `<strong>${slope.toFixed(4)}%/day</strong>, the battery is projected to reach end-of-life in ` +
      `approximately ${rulDays.toLocaleString()} days at the current rate of decline.`);
  }

  if (soh != null && soh > 95 && rulDays != null && rulDays < 400) {
    lines.push(`Despite a healthy-looking absolute SoH of <strong>${fmtPct(soh)}</strong>, ` +
      `the <em>rate</em> of decline is what matters here. At ${Math.abs(slope).toFixed(4)}%/day ` +
      `this is among the fastest-degrading vehicles in the fleet. A battery at 97% falling steeply ` +
      `is more at risk than one at 94% with a flat trajectory.`);
  }

  if (rel === "low_r2") {
    lines.push(`⚠ RUL reliability is flagged <strong>low R²</strong> — slope estimation has ` +
      `reduced statistical confidence. Treat this RUL as directional rather than precise.`);
  } else if (rel === "insufficient_data") {
    lines.push(`⚠ <strong>Insufficient charging sessions</strong> were available to fit a reliable ` +
      `slope. The estimate carries higher uncertainty — prioritise inspection.`);
  }

  return lines.join("<br><br>");
}

/* ─── Coef comparison text generator ────────────────────────────────────────── */
function generateCoefComparison(reg, vehCoef, globalCoef) {
  const shared = Object.keys(vehCoef).filter(k => globalCoef[k] != null && vehCoef[k] < 0);
  if (!shared.length) return "No shared negative coefficient features to compare.";

  const ranked = shared
    .map(k => ({ k, v: vehCoef[k], g: globalCoef[k], diff: vehCoef[k] - globalCoef[k] }))
    .sort((a, b) => a.v - b.v);   // most negative vehicle coef first

  const top3     = ranked.slice(0, 3).map(x => `<strong>${formatCoefLabel(x.k)}</strong>`);
  const vehSpec  = ranked.filter(x => x.diff < -0.000005).slice(0, 2);
  const fleetSpec = ranked.filter(x => x.diff > 0.000005).slice(0, 2);

  const fleetTop = Object.entries(globalCoef)
    .filter(([, v]) => v < 0)
    .sort((a, b) => a[1] - b[1])
    .slice(0, 3)
    .map(([k]) => `<strong>${formatCoefLabel(k)}</strong>`);

  let text = `For <strong>${reg}</strong>, the top wear drivers by model weight are ` +
    `${top3.join(", ")}. `;

  if (vehSpec.length) {
    const labels = vehSpec.map(x => `<strong>${formatCoefLabel(x.k)}</strong>`);
    text += `Compared to the typical vehicle in the fleet, this bus is more affected by ` +
      `${labels.join(" and ")} — these are accelerating its battery wear faster than normal. `;
  }

  if (fleetSpec.length) {
    const labels = fleetSpec.map(x => `<strong>${formatCoefLabel(x.k)}</strong>`);
    text += `Across the whole fleet, ${labels.join(" and ")} are the biggest common wear drivers. ` +
      `This vehicle is less exposed to ${fleetSpec.length > 1 ? "these" : "this"} than most, ` +
      `suggesting a different wear pattern. `;
  } else {
    if (fleetTop.length) {
      text += `The fleet's main wear drivers (${fleetTop.slice(0, 2).join(" and ")}) match this vehicle's profile. `;
    }
  }

  text += `A larger bar means that factor has a stronger link to battery health loss — ` +
    `these are signals to watch, not necessarily the root cause.`;

  return text;
}

/* ─── Tier info lookup ────────────────────────────────────────────────────────── */
function getVehicleTierInfo(reg) {
  const t1 = _tiers.tier1.find(v => v.registration_number === reg);
  if (t1) return { tier: 1, label: "TIER 1 — IMMEDIATE ATTENTION", color: "#b91c1c", bg: "#fef2f2", signal: t1.primary_signal || "" };
  const t2 = _tiers.tier2.find(v => v.registration_number === reg);
  if (t2) return { tier: 2, label: "TIER 2 — MONITOR CLOSELY", color: "#92400e", bg: "#fffbeb", signal: t2.note || "" };
  const t3 = _tiers.tier3.find(v => v.registration_number === reg);
  if (t3) return { tier: 3, label: "TIER 3 — ELEVATED BUT STABLE", color: "#166534", bg: "#f0fdf4", signal: t3.note || "" };
  return { tier: 0, label: "", color: "#475569", bg: "#f8fafc", signal: "" };
}

/* ─── Open / Close ───────────────────────────────────────────────────────────── */
async function openVehicleDetail(reg) {
  const vehicle  = (_vehicles || []).find(v => v.registration_number === reg);
  if (!vehicle) return;

  const backdrop = document.getElementById("vdBackdrop");
  const body     = document.getElementById("vdBody");
  const tierInfo = getVehicleTierInfo(reg);

  document.getElementById("vdTitle").textContent = reg;
  document.getElementById("vdTierBadge").textContent = tierInfo.label;
  document.getElementById("vdTierBadge").style.color  = tierInfo.color;
  document.getElementById("vdTierBadge").style.background = tierInfo.bg;

  body.innerHTML = `<div style="text-align:center;padding:48px">
    <div class="spinner-border text-primary"></div><p style="margin-top:12px;color:#64748b">Loading vehicle data…</p>
  </div>`;
  backdrop.style.display = "flex";

  // Prevent page scroll while open
  document.body.style.overflow = "hidden";

  try {
    const [bands, coef, sessData, bdData] = await Promise.all([
      fetch(`/api/soh-bands/${reg}/`).then(r => r.json()),
      fetch(`/api/bayes-coef/${reg}/`).then(r => r.json()),
      fetch(`/api/sessions/${reg}/`).then(r => r.json()),
      fetch(`/api/anomaly-breakdown/${reg}/`).then(r => r.json()),
    ]);

    body.innerHTML = "";
    renderVDSection1(vehicle, tierInfo, body);
    renderVDSection2(bands, vehicle, body);
    renderVDSection3(coef, vehicle, body);
    renderVDSection4(sessData, bdData, reg, body);
    // renderVDSection2b removed (task 7: hot/weak subsystem chart removed from modal)
  } catch (e) {
    body.innerHTML = `<div style="padding:24px;color:#b91c1c">Failed to load data: ${e.message}</div>`;
  }
}

function closeVehicleDetail() {
  document.getElementById("vdBackdrop").style.display = "none";
  document.body.style.overflow = "";
  document.querySelectorAll(".tier-vehicle-row.tier-active").forEach(r => r.classList.remove("tier-active"));
  _vdActiveSessId = null;
}

/* ─── Section 1: Signal Analysis ────────────────────────────────────────────── */
function renderVDSection1(vehicle, tierInfo, container) {
  const soh       = vehicle.current_soh;
  const slope     = vehicle["soh_slope_%per_day"];
  const composite = vehicle.composite_degradation_score;
  const anomCount = vehicle.n_combined_anom;
  const eol       = (_overview && _overview.eol_threshold) || 80;
  const headroom  = soh != null ? (soh - eol).toFixed(2) : "—";

  // Compute RUL using same formula as the RUL scatter chart: (SoH headroom) / |daily slope|
  const rulDays = (soh != null && slope != null && slope < 0)
    ? Math.max(0, Math.round((soh - eol) / Math.abs(slope)))
    : null;

  const rulColor   = rulDays == null ? "#64748b" : rulDays < 180 ? "#ef4444" : rulDays < 365 ? "#f59e0b" : "#22c55e";
  const sohColor   = soh == null ? "#64748b" : soh < 90 ? "#ef4444" : soh < 95 ? "#f59e0b" : "#22c55e";
  const rulDisplay = rulDays != null
    ? `${rulDays.toLocaleString()} days <span style="font-size:.7rem;font-weight:400;color:#94a3b8">(${(rulDays/365.25).toFixed(2)} yr)</span>`
    : "—";

  const rulAnalysis       = generateRulAnalysis(vehicle);
  const compositeAnalysis = generateCompositeAnalysis(vehicle);

  const sec = document.createElement("div");
  sec.className = "vd-section";
  sec.innerHTML = `
    <div class="vd-section-hdr">Signal Analysis</div>
    <div class="vd-stats-grid">
      <div class="vd-stat">
        <div class="vd-stat-label">EKF SoH</div>
        <div class="vd-stat-value" style="color:${sohColor}">${fmtPct(soh)}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">SoH slope</div>
        <div class="vd-stat-value" style="color:${slope < 0 ? '#ef4444' : '#22c55e'}">${slope != null ? slope.toFixed(5) + "%/d" : "—"}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">EKF RUL</div>
        <div class="vd-stat-value" style="color:${rulColor}">${rulDisplay}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">SoH headroom to EoL (${eol}%)</div>
        <div class="vd-stat-value">${headroom}%</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">Composite score</div>
        <div class="vd-stat-value">${composite != null ? composite.toFixed(4) : "—"}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">Anomalies</div>
        <div class="vd-stat-value">${anomCount != null ? anomCount : "—"}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">Bayes SoH pred.</div>
        <div class="vd-stat-value">${vehicle.bayes_soh_pred != null ? fmtPct(vehicle.bayes_soh_pred) : "—"}</div>
      </div>
      <div class="vd-stat">
        <div class="vd-stat-label">Reliability</div>
        <div class="vd-stat-value" style="font-size:.8rem">${vehicle.rul_reliability ? vehicle.rul_reliability.replace(/_/g," ") : "—"}</div>
      </div>
    </div>
    <div class="vd-analysis-box">${tierInfo.signal || "No signal note available."}</div>
    ${rulAnalysis       ? `<div class="vd-rul-warn">${rulAnalysis}</div>` : ""}
    ${compositeAnalysis ? `<div class="vd-rul-warn" style="background:#fffbeb;border-color:#f59e0b;color:#78350f;margin-top:10px">${compositeAnalysis}</div>` : ""}
    <div id="vdRulScatterWrap" style="margin-top:14px"></div>
  `;
  container.appendChild(sec);

  // Render RUL-vs-time scatter after DOM paint
  requestAnimationFrame(() => _renderRulScatter(vehicle.registration_number));
}

/* ─── Section 2: EKF Bollinger Bands + Slope ─────────────────────────────────── */
function renderVDSection2(d, vehicle, container) {
  const sec = document.createElement("div");
  sec.className = "vd-section";

  if (d.error || !d.bands || !d.bands.length) {
    sec.innerHTML = `<div class="vd-section-hdr">EKF SoH Bollinger Bands</div>
      <div style="padding:20px;color:#94a3b8;font-size:.85rem">${d.error || "No EKF band data available."}</div>`;
    container.appendChild(sec);
    return;
  }

  const chartId = "vdBandsChart";
  sec.innerHTML = `<div class="vd-section-hdr">EKF SoH Bollinger Bands</div>
    <div id="${chartId}" style="min-height:320px"></div>`;
  container.appendChild(sec);

  // Render after DOM paint
  requestAnimationFrame(() => {
    const dates = d.bands.map(b => b.date);
    const ekf   = d.bands.map(b => b.ekf_soh);
    const upper = d.bands.map(b => Math.min(100, b.upper));   // cap at 100%
    const lower = d.bands.map(b => b.lower);
    const bms   = d.bands.map(b => b.bms_soh_obs);

    const traces = [
      { x: dates, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip", name: "_lower" },
      { x: dates, y: upper, type: "scatter", mode: "lines",
        fill: "tonexty", fillcolor: "rgba(59,130,246,0.10)",
        line: { color: "rgba(99,102,241,0.45)", dash: "dot", width: 1 },
        name: "±2σ band", hovertemplate: "Upper: %{y:.2f}%<extra></extra>" },
      { x: dates, y: ekf, type: "scatter", mode: "lines",
        line: { color: "#3b82f6", width: 2.5 },
        name: "EKF SoH", hovertemplate: "EKF SoH: %{y:.3f}%<extra></extra>" },
    ];

    // BMS obs
    const bmsClean = bms.filter(v => v != null && isFinite(v));
    if (bmsClean.length) {
      traces.push({ x: dates, y: bms, type: "scatter", mode: "markers",
        marker: { color: "#f59e0b", size: 5, opacity: 0.65 },
        name: "BMS SoH", hovertemplate: "BMS: %{y:.1f}%<extra></extra>" });
    }

    // Discharge dots
    if (d.discharge_ekf && d.discharge_ekf.length) {
      traces.push({ x: d.discharge_ekf.map(b => b.date), y: d.discharge_ekf.map(b => b.ekf_soh),
        type: "scatter", mode: "markers",
        marker: { color: "#94a3b8", size: 3, opacity: 0.5 },
        name: "EKF discharge", hovertemplate: "Discharge: %{y:.3f}%<extra></extra>" });
    }

    // Slope trend line anchored at first EKF value
    const slope = vehicle["soh_slope_%per_day"];
    if (slope != null && dates.length >= 2) {
      const d0    = new Date(dates[0]);
      const d1    = new Date(dates[dates.length - 1]);
      const nDays = (d1 - d0) / 86400000;
      const y0    = ekf[0];
      const y1    = y0 + slope * nDays;
      traces.push({ x: [dates[0], dates[dates.length - 1]], y: [y0, y1],
        type: "scatter", mode: "lines",
        line: { color: slope < 0 ? "#ef4444" : "#22c55e", width: 1.5, dash: "dash" },
        name: `Slope (${slope.toFixed(4)}%/d)`,
        hovertemplate: "Trend: %{y:.3f}%<extra></extra>" });
    }

    const bandY = [...ekf, ...upper, ...lower].filter(v => v != null && isFinite(v));
    const yMin  = Math.floor(Math.min(...bandY) - 0.5);
    const yMax  = Math.ceil(Math.max(...bandY)  + 0.5);

    Plotly.newPlot(chartId, traces, {
      paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
      font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 11, color: "#475569" },
      margin: { l: 55, r: 20, t: 20, b: 55 },
      xaxis: { title: "Date", gridcolor: "#e2e8f0", tickangle: -30 },
      yaxis: { title: "SoH (%)", gridcolor: "#e2e8f0", range: [yMin, yMax] },
      legend: { orientation: "h", y: -0.22, font: { size: 10 } },
      hovermode: "x unified",
    }, { displayModeBar: false, responsive: true });
  });
}

/* ─── Section 2b: Hot / Weak Subsystem consistency time-series ───────────────── */
function renderVDSection2b(sessData, container) {
  if (!sessData || !sessData.sessions || !sessData.sessions.length) return;

  const sessions = [...sessData.sessions]
    .filter(s => s.weak_subsystem_consistency != null || s.hot_subsystem_consistency != null)
    .sort((a, b) => (a.start_time_ist || "") < (b.start_time_ist || "") ? -1 : 1);

  if (!sessions.length) return;

  const sec = document.createElement("div");
  sec.className = "vd-section";
  const chartId = "vdSubsysChart";
  sec.innerHTML = `
    <div class="vd-section-hdr">Hot &amp; Weak Subsystem Consistency</div>
    <div style="font-size:.76rem;color:#64748b;margin-bottom:8px;line-height:1.5">
      How consistently the same subsystem stays the hottest (🔴) or weakest (🔵) across sessions.
      A score near 1 = same group is always the problem. Near 0 = the problem jumps around (less actionable).
      Persistently low scores with recurring anomaly flags warrant subsystem-level inspection.
    </div>
    <div id="${chartId}" style="min-height:260px"></div>
  `;
  container.appendChild(sec);

  requestAnimationFrame(() => {
    const dates = sessions.map(s => s.start_time_ist || "");
    const weak  = sessions.map(s => s.weak_subsystem_consistency ?? null);
    const hot   = sessions.map(s => s.hot_subsystem_consistency  ?? null);

    const traces = [
      { type: "scatter", mode: "lines+markers",
        x: dates, y: weak,
        line: { color: "#3b82f6", width: 1.5 }, marker: { size: 3.5 },
        name: "Weak subsystem consistency",
        hovertemplate: "%{x}<br>Weak: %{y:.3f}<extra></extra>" },
      { type: "scatter", mode: "lines+markers",
        x: dates, y: hot,
        line: { color: "#ef4444", width: 1.5 }, marker: { size: 3.5 },
        name: "Hot subsystem consistency",
        hovertemplate: "%{x}<br>Hot: %{y:.3f}<extra></extra>" },
    ];

    Plotly.newPlot(chartId, traces, {
      paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
      font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
      margin: { l: 55, r: 16, t: 16, b: 60 },
      xaxis: { gridcolor: "#e2e8f0", tickangle: -30, tickfont: { size: 8 } },
      yaxis: { title: { text: "Consistency (0–1)", font: { size: 9 } }, gridcolor: "#e2e8f0", range: [0, 1.05] },
      showlegend: true,
      legend: { orientation: "h", y: -0.25, font: { size: 9.5 } },
      hovermode: "x unified",
    }, { displayModeBar: false, responsive: true });
  });
}

/* ─── Section 3: Bayesian Ridge Coefficients ─────────────────────────────────── */
function renderVDSection3(coef, vehicle, container) {
  const sec = document.createElement("div");
  sec.className = "vd-section";

  const hasVeh    = coef.vehicle && Object.keys(coef.vehicle).length > 0;
  const hasGlobal = coef.global  && Object.keys(coef.global).length  > 0;

  if (!hasVeh && !hasGlobal) {
    sec.innerHTML = `<div class="vd-section-hdr">Bayesian Ridge Coefficients</div>
      <div style="padding:20px;color:#94a3b8;font-size:.85rem">${coef.error || "No coefficient data available."}</div>`;
    container.appendChild(sec);
    return;
  }

  const vChartId = "vdCoefVehicle";
  const gChartId = "vdCoefGlobal";

  sec.innerHTML = `
    <div class="vd-section-hdr">Bayesian Ridge Coefficients</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div style="font-size:.72rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">
          ${vehicle.registration_number} — per-vehicle
        </div>
        <div id="${vChartId}"></div>
      </div>
      <div>
        <div style="font-size:.72rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">
          Fleet global (median)
        </div>
        <div id="${gChartId}"></div>
      </div>
    </div>
    <div class="vd-coef-compare" id="vdCoefCompare" style="margin-top:14px">
      Generating analysis…
    </div>
  `;
  container.appendChild(sec);

  requestAnimationFrame(() => {
    _renderCoefBar(vChartId, coef.vehicle,  "#3b82f6");
    _renderCoefBar(gChartId, coef.global,   "#6366f1");

    if (hasVeh && hasGlobal) {
      document.getElementById("vdCoefCompare").innerHTML =
        generateCoefComparison(vehicle.registration_number, coef.vehicle, coef.global);
    }
  });
}

function _renderCoefBar(divId, coefObj, color) {
  const entries = Object.entries(coefObj)
    .filter(([, v]) => v < 0)
    .sort((a, b) => a[1] - b[1]);  // most negative first (will be longest bar)

  if (!entries.length) {
    document.getElementById(divId).innerHTML =
      `<div style="padding:16px;color:#94a3b8;font-size:.82rem">No negative coefficients.</div>`;
    return;
  }

  const labels = entries.map(([k]) => formatCoefLabel(k));
  const values = entries.map(([, v]) => v);   // keep negative; reversed axis makes bars go right
  const h      = Math.max(220, entries.length * 24 + 70);

  document.getElementById(divId).style.minHeight = h + "px";
  Plotly.newPlot(divId, [{
    type: "bar", orientation: "h",
    y: labels, x: values,
    marker: { color },
    hovertemplate: "%{y}<br>coef: %{x:.6f}<extra></extra>",
  }], {
    paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    height: h,
    margin: { l: 170, r: 16, t: 16, b: 36 },
    xaxis: { title: "Coefficient value", gridcolor: "#e2e8f0", tickfont: { size: 9 },
             autorange: "reversed" },   // negative values + reversed → bars extend left→right
    yaxis: { automargin: true, tickfont: { size: 9.5 } },
  }, { displayModeBar: false, responsive: true });
}

/* ─── Composite score explanation ────────────────────────────────────────────── */
function generateCompositeAnalysis(vehicle) {
  const soh       = vehicle.current_soh;
  const slope     = vehicle["soh_slope_%per_day"];
  const composite = vehicle.composite_degradation_score;
  if (composite == null || slope == null) return "";

  // Only show when there's a notable disconnect: fast slope but low composite
  const fastSlope   = slope < -0.08;
  const lowComposite = composite < 0.40;
  if (!fastSlope || !lowComposite) return "";

  const eol      = (_overview && _overview.eol_threshold) || 80;
  const deficit  = soh != null ? (100 - soh).toFixed(2) : "?";

  return `<strong>Why is the composite score low despite such a steep slope?</strong><br><br>` +
    `The composite score (${composite.toFixed(4)}) is a weighted combination of <em>seven signals</em>, ` +
    `with the SoH slope carrying <strong>zero direct weight</strong>. The primary component (25%) is the ` +
    `EKF SoH <em>health deficit</em> — how far current SoH sits below the fleet maximum, not how fast ` +
    `it is falling. At ${fmtPct(soh)}, the health deficit is only ${deficit}%, which is low relative to ` +
    `the fleet, keeping that component close to zero. The remaining 75% weights secondary signal ` +
    `<em>trend slopes</em>: rising V-sag rate (15%), IR event rate (15%), energy/km (13%), temp rise ` +
    `rate (11%), and cell spread (11%). If these secondary signals are currently flat or near-fleet-average ` +
    `for this vehicle, the composite stays low regardless of how fast SoH is declining.<br><br>` +
    `<strong>Could the slope be exaggerated (omitted variable bias)?</strong><br><br>` +
    `Possibly. The EKF slope is estimated from a finite window of charging sessions. Several factors can ` +
    `inflate the apparent rate: (1) <em>seasonal temperature effects</em> — cold ambient temperatures ` +
    `suppress observed capacity, creating an apparent downward trend if the data span crosses seasons; ` +
    `(2) <em>sparse early data</em> — if the first few charging sessions had high EKF uncertainty or ` +
    `atypically high BMS-reported SoH, the OLS anchor is too high, steepening the fitted line; ` +
    `(3) <em>charging depth shifts</em> — if the vehicle shifted from consistent full charges to ` +
    `partial charges, observed SoH comparison points become inconsistent; ` +
    `(4) <em>BMS calibration drift</em> — periodic BMS recalibrations can introduce step-changes that ` +
    `look like gradual decline in aggregate. The RUL estimate should be treated as a best-available ` +
    `forward projection, not a precise countdown — increase monitoring frequency and validate with ` +
    `a controlled full-charge capacity test.`;
}

/* ─── Section 4: Anomalous Sessions ─────────────────────────────────────────── */
/* ════════════════════════════════════════════════════════════════════════════
   SECTION 4 — ALERT BREAKDOWN + ANOMALOUS SESSIONS + TELEMETRY
   (exact replication of localhost sections 7 / 8 / 9)
   ════════════════════════════════════════════════════════════════════════════ */

// ── Session table constants ────────────────────────────────────────────────
const VD_SESS_HEADERS = [
  "Start","End","Calendar Aging","Type","Start SoC","End SoC","SoC Diff","BMS SoH","EKF SoH","Duration (hrs)",
  "IF Score","IF Anomaly","CUSUM Anomaly","Degr. Score","Reason","Flagged",
  "V-Sags","IR Mean","Spread","Energy/km","Energy kWh","Low SOC",
  "Ref Cap AH","Voltage","Current","Cap AH Dischrg.","Cap AH Regen","Cap AH Plugin","Cap AH Chrg. Total",
  "Cycle SOH","Block Cap AH","Block Odm Km","Session Odo Km","Chg Rate KW",
  "Cell Spread","Weak Subsys.","Hot Subsys.","Subsys V Std",
  "Temp Rise","BMS Cov.","Speed","Is Loaded","Cum EFC",
  "Aging Index",
  "VSag Rate/hr","IR Event Rate","IR EWM10","Spread EWM10",
  "Temp EWM10","VSag EWM10","VSag Trend","IR Evt Trend",
  "IR Trend","Spread Trend","SoH Trend","C-Rate Chg",
  "DoD Stress","Thermal Stress","E/Loaded","Total Alerts",
  "Cell Health Poor","Cell Undervolt","Cell Overvolt","Rapid Heat","High E/km","Slow Chg","Fast Chg",
];
const VD_SESS_FIELDS = [
  "start_time_ist","end_time_ist","days_since_first","session_type","soc_start","soc_end","soc_diff","soh","ekf_soh","duration_hr",
  "if_score","if_anomaly","cusum_anomaly","composite_degradation_score","anomaly_reason","is_anomalous",
  "n_vsag","ir_ohm_mean","cell_spread_mean","energy_per_km","energy_kwh","n_low_soc",
  "ref_capacity_ah","voltage_mean_new","current_mean_new","capacity_ah_discharge_new","capacity_ah_charge_new","capacity_ah_plugin_new","capacity_ah_charge_total_new",
  "cycle_soh","block_capacity_ah","block_odometer_km","odometer_km","charging_rate_kw",
  "cell_spread_max","weak_subsystem_consistency","hot_subsystem_consistency","subsystem_voltage_std",
  "temp_rise_rate","bms_coverage","speed_mean","is_loaded","cum_efc",
  "aging_index",
  "vsag_rate_per_hr","ir_event_rate","ir_ohm_mean_ewm10","cell_spread_mean_ewm10",
  "temp_rise_rate_ewm10","vsag_rate_per_hr_ewm10","vsag_trend_slope","ir_event_trend_slope",
  "ir_ohm_trend_slope","spread_trend_slope","soh_trend_slope","c_rate_chg",
  "dod_stress","thermal_stress","energy_per_loaded_session","total_alerts",
  "cell_health_poor","n_cell_undervoltage","n_cell_overvoltage","rapid_heating","high_energy_per_km","slow_charging","fast_charging",
];
const VD_SESS_BOOL = new Set(["if_anomaly","cusum_anomaly","is_anomalous",
  "cell_health_poor","rapid_heating","high_energy_per_km","slow_charging","fast_charging"]);

// ── Column descriptions for (i) icons ─────────────────────────────────────
const VD_COL_DESCRIPTIONS = {
  "Vehicle":          "Vehicle registration number",
  "Calendar Aging":   "Days since this vehicle's first recorded session (calendar days, not cycle count)",
  "Start":            "Session start time (IST)",
  "End":              "Session end time (IST)",
  "Type":             "Charging or discharging (operational) session",
  "Start SoC":        "Battery charge level at session start (%)",
  "End SoC":          "Battery charge level at session end (%)",
  "SoC Diff":         "Change in battery charge level this session (Start SoC − End SoC, %). Positive = battery discharged; negative = battery charged up",
  "BMS SoH":          "Battery health as reported by the onboard Battery Management System (%)",
  "EKF SoH":          "Battery health estimated by our real-time tracking model — more accurate than BMS reporting (%)",
  "Duration (hrs)":   "Total session duration in hours",
  "IF Score":         "Abnormality score — higher values mean this session was statistically more unusual",
  "IF Anomaly":       "Flagged by outlier detection (unusual combination of signals in this session)",
  "CUSUM Anomaly":    "Flagged by drift tracker (sustained deterioration detected across multiple sessions)",
  "Degr. Score":      "Combined degradation risk score (0–100 scale, higher = worse). Weighted combination of SoH health deficit, IR trend, cell spread, V-sag rate, temperature rise rate, and energy/km signals",
  "Reason":           "Primary signal(s) that triggered the anomaly flag",
  "Flagged":          "Whether this session is marked as anomalous",
  "V-Sags":           "Count of voltage dip events during this session — high counts indicate the battery is under electrical stress",
  "IR Mean":          "Average internal resistance during this session (Ω) — rising values indicate electrochemical wear inside the cells",
  "Spread":           "Average voltage gap between the strongest and weakest cell group (mV) — higher means more uneven aging across the pack",
  "Energy/km":        "Energy consumed per kilometre driven in this session (kWh/km) — higher means lower efficiency",
  "Energy kWh":       "Total energy consumed or delivered in this session (kWh)",
  "Low SOC":          "Number of times the battery dropped to a critically low charge level during this session",
  "Ref Cap AH":       "Reference capacity of the battery pack (Ah) — used as denominator for SoH calculations",
  "Voltage":          "Average pack voltage during the session (V) — from the hves1 voltage sensor (more accurate than BMS)",
  "Current":          "Average current during the session (A) — from the hves1_current sensor (more accurate than BMS internal current measurement)",
  "Cap AH Dischrg.":    "Ah drawn from the pack during motoring (hves1_current > 0 while vehicle moving). Unit: Ah",
  "Cap AH Regen":       "Ah recovered via regenerative braking (hves1_current < 0 while vehicle moving). Unit: Ah",
  "Cap AH Plugin":      "Ah pushed into the pack during plug-in charging (hves1_current < 0 while vehicle stationary). Unit: Ah",
  "Cap AH Chrg. Total": "Total charge Ah this session = regen + plugin Ah. Used for charging-side SoH and block capacity. Unit: Ah",
  "Cycle SOH":          "Battery health estimated by Ah integration (coulomb counting) over this drive/charge cycle (%). Compares Ah throughput against rated capacity. See FAQ for details",
  "Block Cap AH":       "Total Ah across the entire drive-to-charge block (discharge block = motoring Ah; charge block = regen + plugin Ah). Unit: Ah",
  "Block Odm Km":     "Total distance covered in this drive-to-charge block (km) — cumulative across all sessions in the block, not this session alone",
  "Session Odo Km":   "Distance driven in this individual session (km) — from the VCU odometer reading for this trip only",
  "Chg Rate KW":      "Average charging power during this session (kW) — derived from hves1 current × voltage",
  "Cell Spread":      "Maximum cell voltage imbalance this session (mV) — higher = more uneven aging between cells",
  "Weak Subsys.":     "How consistently the weakest subsystem (lowest voltage group) stays weak (0–1). Lower = the weakest subsystem changes frequently, suggesting measurement noise or fluctuating cell states rather than a clearly identified weak group",
  "Hot Subsys.":      "How consistently the hottest subsystem (highest temperature group) stays hot (0–1). Lower = the hottest location changes between readings, indicating distributed thermal behaviour rather than a persistent hotspot",
  "Subsys V Std":     "Standard deviation of mean voltages across subsystems during this session (V). High = subsystems running at noticeably different voltages — indicates cell imbalance between groups",
  "Temp Rise":        "Rate of temperature rise in the battery pack (highest temperature sensor) during this session (°C/min) — sustained high rates indicate thermal stress",
  "BMS Cov.":         "Fraction of the session with valid BMS data (0–1). Low values mean gaps in telemetry",
  "Speed":            "Average vehicle speed during the session (km/h)",
  "Is Loaded":        "Whether the vehicle carried passengers/cargo — Inbound = loaded (is_loaded=1); Outbound = empty (is_loaded=0)",
  "Cum EFC":          "Cumulative equivalent full charge cycles since vehicle entered service (count)",
  "Aging Index":      "Composite aging score combining calendar age (days), cycle count (EFC), and usage intensity",
  "VSag Rate/hr":     "Rate of voltage sag events per hour of operation (events/hr)",
  "IR Event Rate":    "Rate of high-IR events per session (events/session)",
  "IR EWM10":         "Exponentially-weighted moving average of IR over the last 10 sessions (Ω) — smoothed trend",
  "Spread EWM10":     "Exponentially-weighted moving average of cell spread over last 10 sessions (mV) — smoothed trend",
  "Temp EWM10":       "Exponentially-weighted moving average of temperature rise rate over last 10 sessions (°C/min)",
  "VSag EWM10":       "Exponentially-weighted moving average of voltage sag rate over last 10 sessions (events/hr)",
  "VSag Trend":       "Long-term slope of voltage sag rate (events/hr per session) — positive means worsening",
  "IR Evt Trend":     "Long-term slope of IR event rate (events/session per session) — positive means worsening",
  "IR Trend":         "Long-term slope of internal resistance (Ω per session) — positive means rising resistance",
  "Spread Trend":     "Long-term slope of cell spread (mV per session) — positive means increasing imbalance",
  "SoH Trend":        "Long-term slope of battery health (%/session) — negative means declining SoH",
  "C-Rate Chg":       "Charging rate relative to battery capacity (C) — higher C-rate means faster charging, which accelerates degradation",
  "DoD Stress":       "Depth of discharge stress index — high values mean deep cycling (large SoC swings, harder on battery chemistry)",
  "Thermal Stress":   "Thermal stress index — combines peak temperature and duration of elevated temperature exposure. See FAQ for calculation details",
  "E/Loaded":         "Energy consumed per loaded (passenger-carrying) session (kWh) — normalised for loaded trips only",
  "Total Alerts":     "Total number of BMS alerts raised during this session (count)",
  "Cell Health Poor": "Whether any cell was flagged as in poor health (Yes/No)",
  "Cell Undervolt":   "Number of cells that fell below the minimum allowable voltage threshold during this session",
  "Cell Overvolt":    "Number of cells that exceeded the maximum allowable voltage threshold during this session",
  "Rapid Heat":       "Whether the battery heated up unusually fast (Yes/No) — flag triggered when temperature rise rate exceeds the fleet 95th percentile",
  "High E/km":        "Whether energy consumption per km was unusually high (Yes/No) — flag triggered when efficiency drops below fleet norm",
  "Slow Chg":         "Whether charging was unusually slow (Yes/No) — possible charger issue or battery impedance increase",
  "Fast Chg":         "Whether fast (DC) charging was used in this session (Yes/No)",
};

// ── Per-overlay state ─────────────────────────────────────────────────────
let _vdSessCache    = null;
let _vdSessFilter   = { detector: null, signal: null, sessionType: null };
let _vdReg          = null;
let _vdBdCache      = null;   // stored so detector chart can re-render itself on click
let _vdActiveSessId = null;   // session_id of currently-open telemetry row

// ── CUSUM / IF filter maps (mirrors dashboard.js) ─────────────────────────
const VD_CUSUM_FILTER = {
  "EKF SoH Decline": s => !!(s.cusum_ekf_soh_alarm),
  "BMS SoH Decline": s => !!(s.cusum_soh_alarm),
  "Cycle SoH Drop":  s => !!(s.cusum_cycle_soh_alarm),
  "IR Degradation":  s => !!(s.cusum_ir_slope_alarm) || (s.n_high_ir > 0),
  "Cell Spread":     s => !!(s.cusum_spread_alarm) || !!(s.cusum_spread_slope_alarm),
  "Thermal Stress":  s => !!(s.cusum_heat_alarm),
  "Efficiency Loss": s => !!(s.cusum_epk_alarm),
  "Voltage Sag":     s => (s.n_vsag > 0),
};
const VD_IF_KEYWORDS = {
  "IR Degradation":          ["n_high_ir","ir_ohm_mean","ir_event_rate"],
  "Voltage Sag":             ["n_vsag","d_vsag_per_cycle"],
  "Cell Spread / Imbalance": ["cell_spread","subsystem_voltage_std"],
  "Thermal Stress":          ["temp","thermal_stress"],
  "Efficiency / Capacity":   ["energy_per_loaded_session","capacity_ah_discharge"],
  "High DoD":                ["dod_stress"],
  "Low SoC / Undervoltage":  ["n_low_soc","voltage_min"],
  "SoH Decline":             ["soh","ekf_soh_delta","cycle_soh"],
  "Usage Pattern":           ["odometer_km","duration_hr"],
};

function _vdApplyFilter(sessions, { excludeSessionType = false } = {}) {
  const { detector, signal, sessionType } = _vdSessFilter;
  let f = sessions;
  if (detector === "if")    f = f.filter(s => !!(s.if_anomaly));
  else if (detector === "cusum") f = f.filter(s =>
    !!(s.cusum_ekf_soh_alarm)||!!(s.cusum_soh_alarm)||!!(s.cusum_cycle_soh_alarm)||
    !!(s.cusum_heat_alarm)||!!(s.cusum_spread_alarm)||!!(s.cusum_spread_slope_alarm)||
    !!(s.cusum_epk_alarm)||!!(s.cusum_ir_slope_alarm));
  if (signal) {
    if (detector === "if") {
      const kws = VD_IF_KEYWORDS[signal] || [];
      f = f.filter(s => kws.some(k => (s.if_reason||"").toLowerCase().includes(k.toLowerCase())));
    } else {
      const fn = VD_CUSUM_FILTER[signal];
      if (fn) f = f.filter(fn);
    }
  }
  if (!excludeSessionType && sessionType) f = f.filter(s => s.session_type === sessionType);
  return f;
}

// ── Donut chart helpers ───────────────────────────────────────────────────
const VD_DONUT_W = 286, VD_DONUT_H = 286;
const VD_DONUT_LAYOUT = {
  paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
  font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
  margin: { l: 10, r: 10, t: 10, b: 10 },
  showlegend: true,
  legend: { orientation: "v", x: 0.72, xanchor: "left", y: 0.5, yanchor: "middle", font: { size: 8.5 }, tracegroupgap: 4 },
};
const VD_SIG_PALETTE = ["#6366f1","#0ea5e9","#10b981","#f59e0b","#ec4899","#8b5cf6","#14b8a6","#94a3b8"];

async function _vdRenderDetectorChart(byDetector) {
  const el = document.getElementById("vdDetChart");
  if (!el) return;
  el.innerHTML = "";
  const detKeys   = Object.keys(byDetector);
  const detValues = Object.values(byDetector);
  const labelToKey = label => label.toLowerCase().includes("isolation") ? "if" : "cusum";
  const pull = detKeys.map(k => labelToKey(k) === _vdSessFilter.detector ? 0.12 : 0);

  await Plotly.newPlot("vdDetChart", [{
    type: "pie", hole: 0.5,
    domain: { x: [0.0, 0.68], y: [0.05, 0.95] },
    labels: detKeys, values: detValues, pull,
    marker: { colors: ["#6366f1","#0ea5e9"] },
    textinfo: "percent", textposition: "inside", insidetextorientation: "radial",
    hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
  }], { ...VD_DONUT_LAYOUT, width: VD_DONUT_W, height: VD_DONUT_H },
  { displayModeBar: false });

  el.removeAllListeners("plotly_click");
  el.on("plotly_click", async data => {
    const label    = detKeys[data.points[0].pointNumber];
    const detector = labelToKey(label);
    _vdSessFilter.detector = (_vdSessFilter.detector === detector) ? null : detector;
    _vdSessFilter.signal   = null;
    // Re-render with updated pull
    await _vdRenderDetectorChart(byDetector);
    const params = new URLSearchParams();
    if (_vdSessFilter.detector)    params.set("detector",     _vdSessFilter.detector);
    if (_vdSessFilter.sessionType) params.set("session_type", _vdSessFilter.sessionType);
    const url = `/api/anomaly-breakdown/${_vdReg}/` + (params.toString() ? "?" + params : "");
    const fd  = await fetch(url).then(r => r.json());
    const parts = [];
    if (_vdSessFilter.detector)    parts.push(label);
    if (_vdSessFilter.sessionType) parts.push(_vdSessFilter.sessionType);
    await _vdRenderSignalChart(fd.by_signal,
      parts.length ? parts.join(" · ") : _vdReg);
    _vdRefreshSessions();
  });
}

async function _vdRenderSignalChart(bySignal, scope) {
  const el = document.getElementById("vdSigChart");
  if (!el) return;
  el.innerHTML = "";
  const filterLabel = scope && scope !== _vdReg ? ` — ${scope}` : "";
  document.getElementById("vdSigHeader").textContent = `Signal Breakdown${filterLabel}`;
  const labels = Object.keys(bySignal), values = Object.values(bySignal);
  if (!labels.length) {
    el.innerHTML = `<div style="padding:20px;color:#94a3b8;font-size:.82rem;text-align:center">No signal data</div>`;
    return;
  }
  const pull = labels.map(l => l === _vdSessFilter.signal ? 0.12 : 0);

  await Plotly.newPlot("vdSigChart", [{
    type: "pie", hole: 0.5,
    domain: { x: [0.0, 0.68], y: [0.05, 0.95] },
    labels, values, pull,
    marker: { colors: VD_SIG_PALETTE.slice(0, labels.length) },
    textinfo: "percent", textposition: "inside", insidetextorientation: "radial",
    hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
  }], { ...VD_DONUT_LAYOUT, width: VD_DONUT_W, height: VD_DONUT_H },
  { displayModeBar: false });

  el.removeAllListeners("plotly_click");
  el.on("plotly_click", data => {
    const signal = labels[data.points[0].pointNumber];
    if (!signal) return;
    _vdSessFilter.signal = (_vdSessFilter.signal === signal) ? null : signal;
    _vdRenderSignalChart(bySignal, scope);  // re-render self to update pull
    _vdRefreshSessions();
  });
}

async function _vdRenderTypeChart(sessions, scope) {
  const el = document.getElementById("vdTypeChart");
  if (!el) return;
  el.innerHTML = "";
  document.getElementById("vdTypeHeader").textContent = `Session Type`;
  if (!sessions.length) {
    el.innerHTML = `<div style="padding:20px;color:#94a3b8;font-size:.82rem;text-align:center">No sessions</div>`;
    return;
  }
  const counts = {};
  sessions.forEach(s => { const t = s.session_type || "unknown"; counts[t] = (counts[t] || 0) + 1; });
  const TYPE_COLORS = { charging: "#f59e0b", discharge: "#6366f1", idle: "#94a3b8" };
  const TYPE_LABELS = { charging: "Charging", discharge: "Discharging", idle: "Idle" };
  const rawKeys = Object.keys(counts);
  const labels  = rawKeys.map(k => TYPE_LABELS[k] || k.charAt(0).toUpperCase() + k.slice(1));
  const values  = Object.values(counts);

  // Pull out the currently-selected slice so the user can see what's active
  const activeST = _vdSessFilter.sessionType;
  const pull = rawKeys.map(k => k === activeST ? 0.12 : 0);

  await Plotly.newPlot("vdTypeChart", [{
    type: "pie", hole: 0.5,
    domain: { x: [0.0, 0.68], y: [0.05, 0.95] },
    labels, values, pull,
    marker: { colors: rawKeys.map(k => TYPE_COLORS[k] || "#6b7280") },
    textinfo: "percent", textposition: "inside", insidetextorientation: "radial",
    hovertemplate: "%{label}: %{value} sessions (%{percent})<extra></extra>",
  }], { ...VD_DONUT_LAYOUT, width: VD_DONUT_W, height: VD_DONUT_H },
  { displayModeBar: false });

  el.removeAllListeners("plotly_click");
  el.on("plotly_click", async data => {
    // Use pointNumber to index into rawKeys — reliable regardless of Plotly's internal slice ordering
    const clicked = rawKeys[data.points[0].pointNumber];
    if (!clicked) return;
    _vdSessFilter.sessionType = (_vdSessFilter.sessionType === clicked) ? null : clicked;

    const params = new URLSearchParams();
    if (_vdSessFilter.detector)    params.set("detector",     _vdSessFilter.detector);
    if (_vdSessFilter.sessionType) params.set("session_type", _vdSessFilter.sessionType);
    const url = `/api/anomaly-breakdown/${_vdReg}/` + (params.toString() ? "?" + params : "");
    const fd  = await fetch(url).then(r => r.json());
    const parts = [];
    if (_vdSessFilter.detector)    parts.push(_vdSessFilter.detector.toUpperCase());
    if (_vdSessFilter.sessionType) parts.push(_vdSessFilter.sessionType);
    await _vdRenderSignalChart(fd.by_signal, parts.length ? `${parts.join(" · ")} — ${_vdReg}` : _vdReg);
    _vdRefreshSessions();
  });
}

// Numeric fields (right-align) — any field that renders as a number
const VD_NUMERIC_FIELDS = new Set([
  "days_since_first","soc_start","soc_end","soc_diff","soh","ekf_soh","duration_hr",
  "if_score","composite_degradation_score",
  "n_vsag","ir_ohm_mean","cell_spread_mean","energy_per_km","energy_kwh","n_low_soc",
  "ref_capacity_ah","voltage_mean_new","current_mean_new",
  "capacity_ah_discharge_new","capacity_ah_charge_new","capacity_ah_plugin_new","capacity_ah_charge_total_new",
  "cycle_soh","block_capacity_ah","block_odometer_km","odometer_km","charging_rate_kw",
  "cell_spread_max","weak_subsystem_consistency","hot_subsystem_consistency","subsystem_voltage_std",
  "temp_rise_rate","bms_coverage","speed_mean","cum_efc","aging_index",
  "vsag_rate_per_hr","ir_event_rate","ir_ohm_mean_ewm10","cell_spread_mean_ewm10",
  "temp_rise_rate_ewm10","vsag_rate_per_hr_ewm10","vsag_trend_slope","ir_event_trend_slope",
  "ir_ohm_trend_slope","spread_trend_slope","soh_trend_slope","c_rate_chg",
  "dod_stress","thermal_stress","energy_per_loaded_session","total_alerts",
  "n_cell_undervoltage","n_cell_overvoltage","block_soc_diff",
]);

// ── Sessions table renderer ───────────────────────────────────────────────
function _vdRenderSessionRows(sessions) {
  if (!sessions.length)
    return `<tr><td colspan="${VD_SESS_HEADERS.length}" style="text-align:center;color:#94a3b8;padding:16px;font-size:.82rem">No sessions match the current filter.</td></tr>`;

  return sessions.map(s => {
    const flagged = isBadAnomaly(s);
    const sid     = s.session_id ?? "";
    const cells   = VD_SESS_FIELDS.map(f => {
      const v = s[f];
      const isNum = VD_NUMERIC_FIELDS.has(f);
      const alignStyle = isNum ? "text-align:right" : "text-align:center";

      if (VD_SESS_BOOL.has(f)) {
        return v ? `<td style="${alignStyle}"><span style="background:#fef3c7;color:#92400e;font-size:.7rem;padding:1px 5px;border-radius:3px;font-weight:600">Yes</span></td>`
                 : `<td style="${alignStyle};color:#cbd5e1;font-size:.75rem">—</td>`;
      }
      if (f === "end_time_ist") return `<td style="white-space:nowrap;text-align:center">${v??'—'}</td>`;
      if (f === "start_time_ist") return `<td style="white-space:nowrap;text-align:center">
        <a class="tier-reg-link" style="font-size:.78rem;font-weight:700;color:#3b82f6;
           text-decoration:underline;text-underline-offset:2px;cursor:pointer;
           display:inline-flex;align-items:center;gap:3px"
           title="Click to view telemetry for this session">
          ${v??'—'}<span style="font-size:.65rem;opacity:.75">↗</span>
        </a></td>`;
      if (f === "registration_number") return `<td style="text-align:center"><span style="font-size:.78rem;font-weight:600">${v??'—'}</span></td>`;
      if (f === "session_type") {
        const label = v==="charging"?"Charging":v==="discharge"?"Discharging":(v??'—');
        const color = v==="charging"?"#fef3c7;color:#92400e":"#eff6ff;color:#1d4ed8";
        return `<td style="text-align:center"><span style="background:${color};font-size:.7rem;padding:1px 5px;border-radius:3px;font-weight:600">${label}</span></td>`;
      }
      if (f === "is_loaded") {
        if (v==null) return `<td style="text-align:center">—</td>`;
        return `<td style="text-align:center">${(v==1||v===true)?"Inbound":"Outbound"}</td>`;
      }
      if (f === "composite_degradation_score") {
        if (v==null) return `<td style="${alignStyle}">—</td>`;
        return `<td style="${alignStyle};font-weight:600">${(v * 100).toFixed(1)}</td>`;
      }
      if (v==null) return `<td style="${alignStyle}">—</td>`;
      if (f==="soc_start"||f==="soc_end") return `<td style="${alignStyle}">${Math.round(v)}%</td>`;
      if (f==="soc_diff") return `<td style="${alignStyle}">${v!=null?v.toFixed(1)+'%':'—'}</td>`;
      if (typeof v==="number") return `<td style="${alignStyle}">${v.toFixed(3)}</td>`;
      if (typeof v==="boolean") return `<td style="${alignStyle}">${v?"True":"False"}</td>`;
      return `<td style="text-align:center">${v}</td>`;
    }).join("");

    const startLbl  = (s.start_time_ist??'').replace(/'/g,'');
    const isActive  = sid && sid === _vdActiveSessId;
    const rowClick  = sid ? `onclick="vdLoadTelemetry('${_vdReg}','${sid}','${s.session_type??''}','${startLbl}')"` : '';
    const titleAttr = sid ? `title="Click to view raw telemetry for this session"` : '';
    const baseBg    = flagged ? '#fffbeb' : '#fff';
    const activeBg  = '#dbeafe';                          // blue-100 when this row is open
    const bgStyle   = isActive ? activeBg : baseBg;
    const activeOutline = isActive
      ? 'outline:2px solid #3b82f6;outline-offset:-2px;' : '';
    return `<tr ${titleAttr} ${rowClick} data-sid="${sid}"
      style="font-size:.72rem;background:${bgStyle};${activeOutline}${sid?'cursor:pointer;':''}transition:background .15s">
      ${cells}
    </tr>`;
  }).join("");
}

function _vdRefreshSessions() {
  if (!_vdSessCache) return;
  const { detector, signal, sessionType } = _vdSessFilter;

  // Pre-type filter (all filters except sessionType) → feeds type chart
  const preType = _vdApplyFilter(_vdSessCache.sessions, { excludeSessionType: true });
  _vdRenderTypeChart(preType, _vdReg);

  // Final filtered — strict match; null/undefined session_type never matches
  const filtered = sessionType
    ? preType.filter(s => s.session_type != null && s.session_type === sessionType)
    : preType;

  let filterNote = "";
  if (detector)    filterNote += detector.toUpperCase();
  if (signal)      filterNote += (filterNote?" · ":"")+signal;
  if (sessionType) filterNote += (filterNote?" · ":"")+sessionType;
  if (filterNote)  filterNote = ` <span style="color:#3b82f6;font-weight:600">[${filterNote}]</span>`;

  const note = `Showing <strong>${filtered.length}</strong> anomalous sessions &nbsp;|&nbsp; ` +
    `${_vdSessCache.total_anomalous} anomalous of ${_vdSessCache.total_sessions} total${filterNote} ` +
    `· <span style="color:#3b82f6;text-decoration:underline">rows are clickable</span> for telemetry`;

  const thead = `<thead><tr>${VD_SESS_HEADERS.map(h => {
    const tip = VD_COL_DESCRIPTIONS[h] || "";
    const icon = tip
      ? ` <span style="cursor:help;color:#94a3b8;font-size:.6rem;position:relative" title="${tip}">ⓘ</span>`
      : "";
    return `<th style="background:#f8fafc;color:#475569;font-size:.7rem;font-weight:600;
     text-transform:uppercase;letter-spacing:.04em;padding:8px 16px;white-space:nowrap;
     border-bottom:1px solid #e2e8f0;position:sticky;top:0;z-index:2">${h}${icon}</th>`;
  }).join("")}</tr></thead>`;

  document.getElementById("vdSessContainer").innerHTML =
    `<div style="overflow-x:auto;max-height:420px;border-radius:6px;border:1px solid #e2e8f0">
      <table class="table table-sm table-bordered summary-table" style="border-collapse:collapse;width:100%;min-width:1200px;margin-bottom:0">
        ${thead}<tbody>${_vdRenderSessionRows(filtered)}</tbody>
      </table>
    </div>
    <div style="padding:6px 4px;font-size:.72rem;color:#94a3b8;margin-top:4px">${note}</div>`;
}

// ── Section 4 orchestrator ────────────────────────────────────────────────
async function renderVDSection4(sessData, bdData, reg, container) {
  _vdReg        = reg;
  _vdSessCache  = sessData;
  _vdSessFilter = { detector: null, signal: null, sessionType: null };

  const sec = document.createElement("div");
  sec.className = "vd-section";
  sec.innerHTML = `
    <div class="vd-section-hdr">Anomalous Sessions</div>

    <!-- Detector descriptions (plain language) -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
      <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 14px">
        <div style="font-size:.72rem;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Outlier Detection <span style="font-weight:400;text-transform:none;letter-spacing:0">(IF)</span></div>
        <div style="font-size:.75rem;color:#334155;line-height:1.5">
          Spots <strong>sudden anomalies</strong> — flags sessions where the combination of charge level,
          temperature, cell balance, and internal resistance looks unusual compared to normal behaviour.
          One bad session in isolation. <em>Think: "this session was abnormal."</em>
        </div>
      </div>
      <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:8px;padding:10px 14px">
        <div style="font-size:.72rem;font-weight:700;color:#7e22ce;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Drift Tracking <span style="font-weight:400;text-transform:none;letter-spacing:0">(CUSUM)</span></div>
        <div style="font-size:.75rem;color:#334155;line-height:1.5">
          Identifies <strong>gradual wear</strong> — detects slow, sustained deterioration building up
          over many sessions, such as steadily rising internal resistance or creeping cell imbalance
          that no single session makes obvious. <em>Think: "this vehicle is on a declining trend."</em>
        </div>
      </div>
    </div>

    <!-- Three donut charts -->
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px">
      <div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden">
        <div id="vdDetHeader" style="padding:8px 12px;font-size:.72rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;background:#f8fafc;border-bottom:1px solid #e2e8f0">Alert Source</div>
        <div id="vdDetChart" style="height:${VD_DONUT_H}px;display:flex;align-items:center;justify-content:center"></div>
      </div>
      <div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden">
        <div id="vdTypeHeader" style="padding:8px 12px;font-size:.72rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;background:#f8fafc;border-bottom:1px solid #e2e8f0">Session Type</div>
        <div id="vdTypeChart" style="height:${VD_DONUT_H}px;display:flex;align-items:center;justify-content:center"></div>
      </div>
      <div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden">
        <div id="vdSigHeader" style="padding:8px 12px;font-size:.72rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em;background:#f8fafc;border-bottom:1px solid #e2e8f0">Signal Breakdown</div>
        <div id="vdSigChart" style="height:${VD_DONUT_H}px;display:flex;align-items:center;justify-content:center"></div>
      </div>
    </div>

    <!-- Sessions table -->
    <div id="vdSessContainer">
      <div style="text-align:center;padding:24px;color:#94a3b8">
        <div class="spinner-border spinner-border-sm text-primary"></div>
        <p style="margin-top:8px;font-size:.82rem">Loading sessions…</p>
      </div>
    </div>

    <!-- Inline telemetry -->
    <div id="vdTelSection" style="display:none;margin-top:20px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <div style="font-size:.78rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.05em">
          Session Telemetry — <span id="vdTelLabel" style="color:#3b82f6;text-transform:none;letter-spacing:0"></span>
        </div>
        <button onclick="document.getElementById('vdTelSection').style.display='none'"
          style="background:none;border:1px solid #e2e8f0;border-radius:5px;padding:2px 10px;
                 font-size:.72rem;cursor:pointer;color:#64748b">Close ✕</button>
      </div>
      <div id="vdTelBody"></div>
    </div>
  `;
  container.appendChild(sec);

  if (!sessData || !sessData.sessions) {
    document.getElementById("vdSessContainer").innerHTML =
      `<div style="padding:16px;color:#94a3b8;font-size:.82rem">No session data available.</div>`;
  }

  // Render charts after DOM paint
  requestAnimationFrame(async () => {
    if (bdData && bdData.by_detector) {
      await _vdRenderDetectorChart(bdData.by_detector);

      // Click handler now lives inside _vdRenderDetectorChart (self-contained)
      await _vdRenderSignalChart(bdData.by_signal, reg);
    }

    if (sessData && sessData.sessions) {
      _vdRefreshSessions();
    }
  });
}

// ── Detect time gaps in telemetry data ────────────────────────────────────
// Returns indices i where a gap exists between ts[i-1] and ts[i]
function detectTelGaps(timestamps, thresholdMs = 5 * 60 * 1000) {
  const gaps = [];
  for (let i = 1; i < timestamps.length; i++) {
    const t0 = new Date(timestamps[i - 1]).getTime();
    const t1 = new Date(timestamps[i]).getTime();
    if (!isNaN(t0) && !isNaN(t1) && (t1 - t0) > thresholdMs) {
      gaps.push(i);   // index of the first point AFTER the gap
    }
  }
  return gaps;
}

// ── Telemetry renderer ────────────────────────────────────────────────────
async function vdLoadTelemetry(reg, sessionId, sessionType, startTime) {
  // Mark the selected row
  _vdActiveSessId = sessionId;
  document.querySelectorAll(`tr[data-sid]`).forEach(r => {
    const active = r.dataset.sid === sessionId;
    const flagged = r.style.background === 'rgb(255, 251, 235)';  // #fffbeb
    r.style.background    = active ? '#dbeafe' : (flagged ? '#fffbeb' : '#fff');
    r.style.outline       = active ? '2px solid #3b82f6' : '';
    r.style.outlineOffset = active ? '-2px' : '';
  });

  const section = document.getElementById("vdTelSection");
  const body    = document.getElementById("vdTelBody");
  const label   = document.getElementById("vdTelLabel");
  const typeLabel = sessionType==="charging"?"Charging":sessionType==="discharge"?"Discharge":(sessionType||"");
  label.textContent = `${reg} · ${startTime||sessionId} (${typeLabel})`;
  body.innerHTML = `<div style="text-align:center;padding:24px">
    <div class="spinner-border spinner-border-sm text-primary"></div>
    <p style="margin-top:8px;font-size:.82rem;color:#64748b">Loading telemetry…</p></div>`;
  section.style.display = "";
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  const d = await fetch(`/api/telemetry/${reg}/${sessionId}/`).then(r => r.json());
  if (d.error) { body.innerHTML = `<div style="padding:12px;color:#b91c1c;font-size:.82rem">${d.error}</div>`; return; }
  if (!d.rows||!d.rows.length) {
    body.innerHTML = `<div style="padding:16px;color:#94a3b8;font-size:.82rem;text-align:center">No telemetry rows for this session.</div>`;
    return;
  }

  const rows = d.rows;
  // Downsample if too many points (keep ≤500 for performance)
  const MAX_PTS = 500;
  const augRows = (() => {
    let base = rows;
    const isCharging = sessionType === "charging";
    if (isCharging) {
      base = base.map(r => ({
        ...r,
        _chg_pwr: (r.hves1_current!=null&&r.hves1_voltage_level!=null)
          ? Math.abs(r.hves1_current*r.hves1_voltage_level)/1000 : null,
      }));
    } else {
      // Discharge: compute rolling energy/km (window=15) from instantaneous power ÷ speed
      const WIN = 15;
      const epkRaw = base.map(r => {
        if (r.hves1_current == null || r.hves1_voltage_level == null ||
            r.speed == null || r.speed < 5) return null;
        const pwr = Math.abs(r.hves1_current * r.hves1_voltage_level) / 1000; // kW
        const epk = pwr / r.speed;  // kWh/km  (kW ÷ km/h = kWh/km)
        return (epk > 0 && epk < 4) ? epk : null;  // clip implausible spikes
      });
      const epkSmoothed = epkRaw.map((_, i) => {
        const slice = epkRaw.slice(Math.max(0, i - WIN + 1), i + 1).filter(v => v != null);
        return slice.length >= 3 ? slice.reduce((s, v) => s + v, 0) / slice.length : null;
      });
      base = base.map((r, i) => ({ ...r, _epk: epkSmoothed[i] }));
    }
    if (base.length > MAX_PTS) {
      const step = Math.ceil(base.length / MAX_PTS);
      base = base.filter((_, i) => i % step === 0);
    }
    return base;
  })();

  const ts         = augRows.map(r => r.ts||r.gps_time);
  const isCharging = sessionType === "charging";
  const gapIndices = detectTelGaps(ts);  // indices i where gap exists before ts[i]

  // Energy summary from session cache
  const sessData = (_vdSessCache?.sessions || []).find(s => String(s.session_id) === String(sessionId));
  const energySummary = sessData
    ? `<div style="display:flex;gap:16px;flex-wrap:wrap;padding:8px 12px;background:#f8fafc;
        border-bottom:1px solid #e2e8f0;font-size:.75rem;color:#475569">
        <span>Energy: <strong style="color:#0f172a">${sessData.energy_kwh != null ? sessData.energy_kwh.toFixed(2)+' kWh' : '—'}</strong></span>
        <span>Efficiency: <strong style="color:#0f172a">${sessData.energy_per_km != null ? sessData.energy_per_km.toFixed(3)+' kWh/km' : '—'}</strong>
          <span style="color:#94a3b8;font-size:.68rem">(same as energy/km)</span></span>
        <span>Duration: <strong style="color:#0f172a">${sessData.duration_hr != null ? sessData.duration_hr.toFixed(2)+' hr' : '—'}</strong></span>
        <span style="font-size:.68rem;color:#94a3b8">${gapIndices.length > 0 ? `⚠ ${gapIndices.length} data gap${gapIndices.length>1?"s":""} detected (dashed lines)` : "No data gaps detected"}</span>
      </div>`
    : "";

  const chartDefs = [
    { title:"SoC (%)",           fields:[{f:"soc",color:"#3b82f6",name:"SoC"}],                          yLabel:"%" },
    { title:"Temperature (°C)",  fields:[{f:"temperature_highest",color:"#ef4444",name:"Max"},
                                          {f:"temperature_lowest",color:"#06b6d4",name:"Min"}],            yLabel:"°C", multi:true },
    { title:"Cell Spread (mV)",  fields:[{f:"cell_spread",color:"#f59e0b",name:"Spread"}],                yLabel:"mV" },
    { title:"IR (Ω)",            fields:[{f:"ir_ohm",color:"#8b5cf6",name:"IR"}],                         yLabel:"Ω", connectgaps:true },
    { title:"Speed (km/h)",      fields:[{f:"speed",color:"#10b981",name:"Speed"}],                       yLabel:"km/h" },
    ...(isCharging  ? [{ title:"Charging Power (kW)",  fields:[{f:"_chg_pwr",color:"#0ea5e9",name:"Chg Power"}], yLabel:"kW" }] : []),
    ...(!isCharging ? [{ title:"Energy / km (kWh/km)", fields:[{f:"_epk",   color:"#f97316",name:"Eff."}],      yLabel:"kWh/km" }] : []),
    { title:"Voltage Sag Flag",  fields:[{f:"_vsag",color:"#ef4444",name:"Sag"}],                         yLabel:"flag", bar:true },
  ];

  const chartIds = chartDefs.map((_, i) => `vdTelChart_${i}`);
  const pairs = [];
  for (let i = 0; i < chartDefs.length; i += 2) {
    const L = chartDefs[i], R = chartDefs[i + 1];
    pairs.push(`<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
      <div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden">
        <div style="padding:6px 12px;font-size:.72rem;font-weight:600;color:#475569;background:#f8fafc;border-bottom:1px solid #e2e8f0">${L.title}</div>
        <div id="${chartIds[i]}" style="height:180px"></div>
      </div>
      ${R ? `<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden">
        <div style="padding:6px 12px;font-size:.72rem;font-weight:600;color:#475569;background:#f8fafc;border-bottom:1px solid #e2e8f0">${R.title}</div>
        <div id="${chartIds[i+1]}" style="height:180px"></div>
      </div>` : '<div></div>'}
    </div>`);
  }
  body.innerHTML = energySummary + pairs.join("");

  const TEL_LAYOUT = {
    paper_bgcolor:"transparent", plot_bgcolor:"#f8fafc",
    font:{family:"Plus Jakarta Sans, system-ui, sans-serif",size:10,color:"#475569"},
    margin:{l:48,r:10,t:10,b:40},
    xaxis:{gridcolor:"#e2e8f0",tickangle:-25,tickfont:{size:8.5}},
    yaxis:{gridcolor:"#e2e8f0",tickfont:{size:9}},
    showlegend:false,
  };

  const syncIds = [];
  // Progressive render: one chart per animation frame to keep UI responsive.
  // Crosshair wiring runs after ALL charts are done (via onAllDone callback).
  function renderChartAt(i, onAllDone) {
    if (i >= chartDefs.length) { onAllDone && onAllDone(); return; }
    const {fields, yLabel, multi, bar, connectgaps} = chartDefs[i];
    const id = chartIds[i];
    const el = document.getElementById(id);
    if (!el) { requestAnimationFrame(() => renderChartAt(i + 1, onAllDone)); return; }

    const activeFields = fields.filter(({f}) => augRows.some(r => r[f] != null));

    const _hexFill = (hex, a) => {
      const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
      return `rgba(${r},${g},${b},${a})`;
    };

    const traces = activeFields.map(({f, color, name}, fi) => {
      if (bar) {
        return { x: ts, y: augRows.map(r => r[f] ?? null),
                 type: "bar", name, marker: {color},
                 hovertemplate: `%{x}<br>${name}: %{y:.3f}<extra></extra>` };
      }
      // Insert null at each gap index so Plotly breaks the solid line there
      const rawY = augRows.map(r => r[f] ?? null);
      const gappedX = [], gappedY = [];
      let prev = 0;
      for (const gi of gapIndices) {
        gappedX.push(...ts.slice(prev, gi),   null);
        gappedY.push(...rawY.slice(prev, gi), null);
        prev = gi;
      }
      gappedX.push(...ts.slice(prev));
      gappedY.push(...rawY.slice(prev));
      // Fill only the first trace per chart to avoid overlapping fills on multi-line charts
      const fillProps = fi === 0
        ? { fill: "tozeroy", fillcolor: _hexFill(color, 0.13) }
        : {};
      return { x: gappedX, y: gappedY,
               type: "scatter", mode: "lines", name,
               line: {color, width: 1.5},
               connectgaps: !!connectgaps,
               hovertemplate: `%{x}<br>${name}: %{y:.3f}<extra></extra>`,
               ...fillProps };
    });

    // Dashed bridge traces spanning each gap — one per field per gap
    if (!bar) {
      for (const {f, color} of activeFields) {
        const rawY = augRows.map(r => r[f] ?? null);
        for (const gi of gapIndices) {
          const y0 = rawY[gi - 1], y1 = rawY[gi];
          if (y0 != null && y1 != null) {
            traces.push({ x: [ts[gi - 1], ts[gi]], y: [y0, y1],
                          type: "scatter", mode: "lines",
                          line: { color, width: 1.5, dash: "dash" },
                          showlegend: false, hoverinfo: "skip" });
          }
        }
      }
    }

    if (!traces.length) {
      el.innerHTML = `<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:.8rem">No data</div>`;
      requestAnimationFrame(() => renderChartAt(i + 1, onAllDone));
      return;
    }

    const layout = {
      ...TEL_LAYOUT,
      showlegend: !!(multi),
      yaxis: { ...TEL_LAYOUT.yaxis, title: { text: yLabel, font: { size: 8.5 } } },
    };
    if (multi) layout.legend = { orientation:"h", x:0.5, xanchor:"center", y:1.12, font:{size:9} };

    Plotly.newPlot(id, traces, layout, {displayModeBar: false, responsive: true});
    if (!bar) syncIds.push(id);

    requestAnimationFrame(() => renderChartAt(i + 1, onAllDone));
  }

  requestAnimationFrame(() => renderChartAt(0, wireCrosshair));

  // Highlight charts whose signal caused the anomaly
  const anomalyReason = (sessData?.sessions?.find(s => String(s.session_id) === String(sessionId))?.if_reason || "").toLowerCase();
  const FIELD_SIGNAL_MAP = [
    { field: "SoC (%)",           keys: ["soc","n_low_soc"] },
    { field: "Temperature (°C)",  keys: ["temp","thermal"] },
    { field: "Cell Spread (mV)",  keys: ["cell_spread","spread"] },
    { field: "IR (Ω)",            keys: ["ir_ohm","ir_event","n_high_ir"] },
    { field: "Voltage Sag Flag",  keys: ["n_vsag","d_vsag"] },
    { field: "Speed (km/h)",      keys: ["odometer_km","speed"] },
    { field: "Charging Power (kW)",  keys: ["c_rate_chg","slow_charging","fast_charging"] },
    { field: "Energy / km (kWh/km)", keys: ["energy_per_km","high_energy_per_km","energy_per_loaded"] },
  ];
  requestAnimationFrame(() => {
    if (!anomalyReason) return;
    FIELD_SIGNAL_MAP.forEach(({ field, keys }) => {
      if (keys.some(k => anomalyReason.includes(k))) {
        const idx = chartDefs.findIndex(c => c.title === field);
        if (idx >= 0) {
          const el = document.getElementById(chartIds[idx]);
          if (el) el.style.outline = "2px solid #ef4444";
        }
      }
    });
  });

  // Synchronized crosshair — runs after all charts are rendered
  function wireCrosshair() {
  syncIds.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.removeAllListeners("plotly_hover");
    el.removeAllListeners("plotly_unhover");
    el.on("plotly_hover", ev => {
      const xv = ev.points[0].x;
      const shape = [
        {type:"line",x0:xv,x1:xv,y0:0,y1:1,yref:"paper",line:{color:"#ef4444",width:1.5,dash:"solid"}},
        {type:"line",x0:xv,x1:xv,y0:0,y1:1,yref:"paper",line:{color:"#fbbf24",width:1.5,dash:"dot"}},
      ];
      syncIds.filter(i=>i!==id).forEach(oid => {
        const oel=document.getElementById(oid);
        if(oel) Plotly.relayout(oel,{shapes:shape});
      });
    });
    el.on("plotly_unhover", () => {
      syncIds.filter(i=>i!==id).forEach(oid => {
        const oel=document.getElementById(oid);
        if(oel) Plotly.relayout(oel,{shapes:[]});
      });
    });
  });
  }  // end wireCrosshair
}    // end vdLoadTelemetry

/* ─── RUL vs Calendar Days scatter with linear fit + Bollinger bands ─────────── */
function _renderRulScatter(reg) {
  const wrap = document.getElementById("vdRulScatterWrap");
  if (!wrap) return;

  fetch(`/api/soh-bands/${reg}/`)
    .then(r => r.json())
    .then(d => {
      if (!d.bands || !d.bands.length) return;

      const bands = d.bands.filter(b => b.ekf_soh != null);
      if (!bands.length) return;

      const vehicle = (_vehicles || []).find(v => v.registration_number === reg);
      const eol = (_overview && _overview.eol_threshold) || 80;
      const slope = vehicle ? vehicle["soh_slope_%per_day"] : null;

      // Compute approximate RUL at each point: RUL_i = (soh_i − eol) / |slope| in days
      const pts = bands.map((b, i) => ({
        date: b.date,
        dayIdx: i,
        rul: slope && slope < 0 ? Math.max(0, (b.ekf_soh - eol) / Math.abs(slope)) : null,
      })).filter(p => p.rul != null && p.rul < 3000);

      if (!pts.length) return;

      const xVals  = pts.map(p => p.dayIdx);
      const yVals  = pts.map(p => p.rul);
      const xDates = pts.map(p => p.date);

      // ── Least-squares solver (Gaussian elimination) ──────────────────────
      function polyFit(xs, ys, deg) {
        const n = xs.length, m = deg + 1;
        const A   = Array.from({length: n}, (_, i) => Array.from({length: m}, (_, j) => Math.pow(xs[i], j)));
        const AT  = A[0].map((_, ci) => A.map(row => row[ci]));
        const ATA = AT.map(row => AT[0].map((_, j) => row.reduce((s,_,k) => s + row[k]*AT[j][k], 0)));
        const ATy = AT.map(row => row.reduce((s, v, k) => s + v * ys[k], 0));
        const mat = ATA.map((row, i) => [...row, ATy[i]]);
        for (let c = 0; c < m; c++) {
          let maxR = c;
          for (let r = c+1; r < m; r++) if (Math.abs(mat[r][c]) > Math.abs(mat[maxR][c])) maxR = r;
          [mat[c], mat[maxR]] = [mat[maxR], mat[c]];
          for (let r = 0; r < m; r++) {
            if (r === c) continue;
            const f = mat[r][c] / mat[c][c];
            for (let k = c; k <= m; k++) mat[r][k] -= f * mat[c][k];
          }
        }
        return mat.map((row, i) => row[m] / row[i]);
      }

      // ── Linear fit (line of best fit) ────────────────────────────────────
      let linTrace = null;
      if (xVals.length >= 3) {
        try {
          const coefs  = polyFit(xVals, yVals, 1);   // [intercept, slope]
          const linY   = xDates.map((_, i) => coefs[0] + coefs[1] * xVals[i]);
          linTrace = {
            type: "scatter", mode: "lines",
            x: xDates, y: linY,
            line: { color: "#ef4444", width: 2, dash: "solid" },
            name: "Linear fit", hoverinfo: "skip",
          };
        } catch {}
      }

      // ── Bollinger bands (rolling mean ± 2σ, window = 20) ─────────────────
      let bbUpperTrace = null, bbLowerTrace = null, bbMidTrace = null;
      if (yVals.length >= 10) {
        const WIN = Math.min(20, Math.floor(yVals.length / 3));
        const means = [], stds = [];
        for (let i = 0; i < yVals.length; i++) {
          const slice = yVals.slice(Math.max(0, i - WIN + 1), i + 1);
          const m = slice.reduce((s, v) => s + v, 0) / slice.length;
          const s = Math.sqrt(slice.reduce((s, v) => s + (v - m) ** 2, 0) / slice.length);
          means.push(m);
          stds.push(s);
        }
        const upper = means.map((m, i) => m + 2 * stds[i]);
        const lower = means.map((m, i) => Math.max(0, m - 2 * stds[i]));

        // Lower band (invisible, for fill reference)
        bbLowerTrace = {
          type: "scatter", mode: "lines",
          x: xDates, y: lower,
          line: { width: 0, color: "transparent" },
          showlegend: false, hoverinfo: "skip", name: "_bb_lower",
        };
        // Upper band (fills down to lower)
        bbUpperTrace = {
          type: "scatter", mode: "lines",
          x: xDates, y: upper,
          fill: "tonexty", fillcolor: "rgba(99,102,241,0.10)",
          line: { color: "rgba(99,102,241,0.35)", dash: "dot", width: 1 },
          name: "Bollinger ±2σ",
          hovertemplate: "Upper: %{y:.0f} days<extra></extra>",
        };
        // Mid line (rolling mean)
        bbMidTrace = {
          type: "scatter", mode: "lines",
          x: xDates, y: means,
          line: { color: "#6366f1", width: 1.2, dash: "dash" },
          name: "Rolling mean",
          hovertemplate: "Mean: %{y:.0f} days<extra></extra>",
        };
      }

      wrap.innerHTML = `
        <div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-top:4px">
          <div style="padding:8px 12px;font-size:.72rem;font-weight:600;color:#475569;
                      text-transform:uppercase;letter-spacing:.05em;background:#f8fafc;border-bottom:1px solid #e2e8f0">
            Remaining Useful Life — How it changes over time
          </div>
          <div id="vdRulScatterChart" style="height:300px"></div>
        </div>`;

      requestAnimationFrame(() => {
        const traces = [];
        if (bbLowerTrace) traces.push(bbLowerTrace, bbUpperTrace);
        traces.push({
          type: "scatter", mode: "markers",
          x: xDates, y: yVals,
          marker: { color: "#3b82f6", size: 4.5, opacity: 0.65 },
          name: "Est. RUL",
          hovertemplate: "%{x}<br>Est. RUL: %{y:.0f} days<extra></extra>",
        });
        if (bbMidTrace) traces.push(bbMidTrace);
        if (linTrace)   traces.push(linTrace);

        Plotly.newPlot("vdRulScatterChart", traces, {
          paper_bgcolor: "transparent", plot_bgcolor: "#f8fafc",
          font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
          margin: { l: 60, r: 16, t: 16, b: 60 },
          xaxis: { title: { text: "Date", font: { size: 9 } }, gridcolor: "#e2e8f0", tickangle: -30, tickfont: { size: 8.5 } },
          yaxis: { title: { text: "Estimated RUL (days)", font: { size: 9 } }, gridcolor: "#e2e8f0" },
          showlegend: true,
          legend: { x: 0.98, xanchor: "right", y: 0.98, font: { size: 9 } },
          hovermode: "x unified",
        }, { displayModeBar: false, responsive: true });
      });
    })
    .catch(() => {});
}

/* ─── Hot / Weak Subsystem fleet chart ───────────────────────────────────────── */
function renderSubsystemChart() {
  const el = document.getElementById("subsystemHeatPlot");
  if (!el || !_vehicles) return;

  // Use vehicle data if it has these fields, else use sessions aggregate from anom data
  // We'll build from scatter data vehicle-level aggregates if available
  // For now, use _vdSessCache if open, else skip
  if (!_scatterData || !_scatterData.points) return;

  // Group by vehicle: compute mean weak/hot consistency
  const byVeh = {};
  // We'll use the vehicles endpoint composite + anomaly for sorting
  (_vehicles || []).forEach(v => {
    byVeh[v.registration_number] = { reg: v.registration_number, composite: v.composite_degradation_score || 0 };
  });

  // Sort by composite descending
  const sorted = Object.values(byVeh).sort((a,b) => b.composite - a.composite).slice(0, 30);
  if (!sorted.length) return;

  // Placeholder — will render when subsystem data is available
  el.innerHTML = `<div style="padding:20px;color:#94a3b8;font-size:.82rem;text-align:center">
    Subsystem chart renders in vehicle detail modal — click a vehicle to see its subsystem consistency.</div>`;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   ANIMATION SYSTEM
   ═══════════════════════════════════════════════════════════════════════════════ */

function initScrollReveal() {
  // ── IntersectionObserver: fire .visible when element enters viewport ──────
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;

      if (el.classList.contains("reveal-stagger")) {
        el.classList.add("visible");
        // KPI cards — add visible + staggered animation-delay
        el.querySelectorAll(".kpi-card").forEach((card, i) => {
          card.style.animationDelay = `${i * 0.07}s`;
          card.style.animationDuration = "0.45s";
          card.style.animationFillMode = "both";
          card.style.animationName = "fadeInScale";
          card.style.animationTimingFunction = "cubic-bezier(.22,.68,0,1.2)";
        });
      } else {
        el.classList.add("visible");
      }

      // Pulse the nearest section-hdr if there is one
      const hdr = el.classList.contains("section-hdr")
        ? el
        : el.querySelector ? el.querySelector(".section-hdr") : null;
      if (hdr && !hdr.classList.contains("hdr-pulse")) {
        hdr.classList.add("hdr-pulse");
        setTimeout(() => hdr.classList.remove("hdr-pulse"), 3500);
      }

      observer.unobserve(el);  // only animate once
    });
  }, {
    threshold: 0.08,
    rootMargin: "0px 0px -40px 0px",
  });

  // Observe all .reveal and .reveal-stagger elements
  document.querySelectorAll(".reveal, .reveal-stagger").forEach(el => observer.observe(el));

  // ── Animate KPI value numbers with a smooth count-up ─────────────────────
  _animateKpiCounters();
}

function _animateKpiCounters() {
  // Wait a tick so values are already set by renderKPICards
  requestAnimationFrame(() => {
    document.querySelectorAll(".kpi-value").forEach(el => {
      el.classList.add("animating");
      el.addEventListener("animationend", () => el.classList.remove("animating"), { once: true });
    });
  });
}

/* ─── Breakdown Timeline ──────────────────────────────────────────────────────── */
let _breakdownRows = [];

function renderBreakdownTimeline(data) {
  if (!data || !data.timeline || !data.timeline.length) return;

  _breakdownRows = data.timeline;  // store for click-through
  const eol = (_overview && _overview.eol_threshold) || 80;

  _drawBreakdownChart(_breakdownRows, eol, null);
  _buildBreakdownTable(_breakdownRows, eol, null);
}

function _drawBreakdownChart(rows, eol, highlightReg) {
  const el = document.getElementById("breakdownTimelinePlot");
  if (!el) return;

  // Exclude vehicles with near-flat slopes whose projected EoL is > 10 years out —
  // they distort the x-axis scale without adding actionable information.
  const MAX_RUL_DAYS = 3650;
  rows = rows.filter(r => r.rul_days != null && r.rul_days <= MAX_RUL_DAYS);

  const TIER_COLOR = { 1: "#ef4444", 2: "#f59e0b", 3: "#22c55e", 0: "#6366f1" };
  const todayStr   = new Date().toISOString().slice(0, 10);

  // Compute RUL + EoL date the same way as renderAnomalyTiers JS
  const rowsWithRul = rows.map(r => {
    let rul = r.rul_days;  // already computed server-side with same formula
    const eolDate = rul != null
      ? (() => {
          const d = new Date(data_refDate || r.ref_date);
          d.setDate(d.getDate() + Math.round(rul));
          return d.toISOString().slice(0, 10);
        })()
      : r.eol_date;
    return { ...r, rul_computed: rul, eol_computed: eolDate };
  });

  const colors = rowsWithRul.map(r =>
    r.registration_number === highlightReg
      ? "#f59e0b"  // highlighted
      : TIER_COLOR[r.tier] || "#6366f1"
  );
  const sizes  = rowsWithRul.map(r => r.registration_number === highlightReg ? 14 : 10);

  Plotly.newPlot(el, [{
    type: "scatter",
    mode: "markers",
    x: rowsWithRul.map(r => r.eol_date || null),
    y: rowsWithRul.map(r => r.registration_number),
    marker: {
      color: colors, size: sizes, symbol: "diamond",
      line: { color: "#fff", width: 1.5 },
    },
    customdata: rowsWithRul.map(r => [
      r.rul_days != null ? Math.round(r.rul_days) : "—",
      r.current_soh != null ? r.current_soh.toFixed(2) : "—",
      r.soh_slope != null ? r.soh_slope.toFixed(4) : "—",
    ]),
    hovertemplate:
      "<b>%{y}</b><br>" +
      "Projected EoL: <b>%{x}</b><br>" +
      "RUL: %{customdata[0]} days<br>" +
      "Current SoH: %{customdata[1]}%<br>" +
      "Daily slope: %{customdata[2]}%/day<extra></extra>",
  }], {
    paper_bgcolor: "white", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    margin: { l: 130, r: 20, t: 16, b: 64 },
    xaxis: {
      title: { text: "Projected End-of-Life Date", font: { size: 9.5 } },
      type: "date", gridcolor: "#e2e8f0", tickfont: { size: 9 }, tickangle: -30,
    },
    yaxis: { autorange: "reversed", gridcolor: "#e2e8f0", tickfont: { size: 9 } },
    shapes: [{
      type: "line", x0: todayStr, x1: todayStr, y0: 0, y1: 1,
      xref: "x", yref: "paper",
      line: { color: "#3b82f6", width: 1.5, dash: "dash" },
    }],
    annotations: [{
      x: todayStr, y: 1.03, xref: "x", yref: "paper",
      text: "Today", showarrow: false,
      font: { size: 8.5, color: "#3b82f6", family: "Plus Jakarta Sans" },
      bgcolor: "rgba(255,255,255,.85)", borderpad: 2,
    }],
    showlegend: false,
  }, { displayModeBar: false, responsive: true });

  // Click-through: clicking a point highlights row in table
  el.on("plotly_click", evt => {
    const pt  = evt.points[0];
    const reg = pt && pt.y;
    if (!reg) return;
    _buildBreakdownTable(_breakdownRows, eol, reg);
    // Scroll to the highlighted row
    const row = document.querySelector(`#breakdownTableBody tr[data-reg="${reg}"]`);
    if (row) row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    // Redraw chart with highlight
    _drawBreakdownChart(_breakdownRows, eol, reg);
  });
}

// Store ref_date for EoL re-computation
let data_refDate = null;

function _buildBreakdownTable(rows, eol, activeReg) {
  const TIER_BADGE = {
    1: `<span style="background:#fef2f2;color:#b91c1c;font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px">T1</span>`,
    2: `<span style="background:#fffbeb;color:#92400e;font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px">T2</span>`,
    3: `<span style="background:#f0fdf4;color:#166534;font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px">T3</span>`,
  };

  document.getElementById("breakdownTableBody").innerHTML = rows.map(r => {
    const isActive  = r.registration_number === activeReg;
    const sohStr    = r.current_soh != null ? r.current_soh.toFixed(2) + "%" : "—";
    const rulStr    = r.rul_days != null
      ? `${Math.round(r.rul_days).toLocaleString()}d (${(r.rul_days / 365.25).toFixed(1)} yr)` : "—";
    const badge     = TIER_BADGE[r.tier] || "";
    const rowStyle  = isActive ? "background:#eff6ff;font-weight:600;" : "";
    return `<tr data-reg="${r.registration_number}"
               style="cursor:pointer;${rowStyle}"
               onclick="openVehicleDetail && openVehicleDetail('${r.registration_number}')">
      <td>${badge}&nbsp;<span style="font-size:.8rem;font-weight:${isActive?700:400}">${r.registration_number}</span></td>
      <td class="text-end" style="font-size:.8rem">${sohStr}</td>
      <td class="text-end" style="font-size:.8rem">${rulStr}</td>
      <td class="text-end" style="font-size:.8rem;font-weight:600">${r.eol_date || "—"}</td>
    </tr>`;
  }).join("");
}

/* ─── Data Distributions ──────────────────────────────────────────────────────── */
function renderDistributions(data) {
  if (!data) return;
  _renderHistogram(
    "cycleSohHistPlot",
    data.cycle_soh || [],
    "Cycle SoH (%)",
    `n = ${(data.cycle_soh_n || 0).toLocaleString()} quality-gated observations`,
    "#3b82f6",
    1.0,
    [85, 102]
  );
  _renderHistogram(
    "blockSohHistPlot",
    data.block_soh || [],
    "Block SoH (%)",
    `n = ${(data.block_soh_n || 0).toLocaleString()} discharge blocks (deduplicated)`,
    "#10b981",
    1.0,
    [85, 102]
  );
}

function _renderHistogram(elId, vals, xLabel, subtitle, color, binSize, xRange) {
  const el = document.getElementById(elId);
  if (!el || !vals.length) return;

  const sorted = [...vals].sort((a, b) => a - b);
  const mu  = vals.reduce((s, v) => s + v, 0) / vals.length;
  const sig = Math.sqrt(vals.reduce((s, v) => s + (v - mu) ** 2, 0) / vals.length);
  const med = sorted[Math.floor(sorted.length / 2)];

  Plotly.newPlot(el, [{
    type: "histogram",
    x: vals,
    autobinx: false,
    xbins: { size: binSize },
    marker: { color, opacity: 0.78 },
    hovertemplate: `${xLabel}: %{x}<br>Sessions: %{y}<extra></extra>`,
  }], {
    paper_bgcolor: "white", plot_bgcolor: "#f8fafc",
    font: { family: "Plus Jakarta Sans, system-ui, sans-serif", size: 10, color: "#475569" },
    margin: { l: 48, r: 14, t: 38, b: 52 },
    title: {
      text: `${subtitle}  ·  µ = ${mu.toFixed(1)}  σ = ${sig.toFixed(1)}  median = ${med.toFixed(1)}`,
      font: { size: 10, color: "#475569" }, x: 0.02, xanchor: "left",
    },
    xaxis: {
      title: { text: xLabel, font: { size: 9.5 } },
      range: xRange || undefined,
      gridcolor: "#e2e8f0", tickfont: { size: 9 },
    },
    yaxis: {
      title: { text: "Sessions", font: { size: 9.5 } },
      gridcolor: "#e2e8f0", tickfont: { size: 9 },
    },
    shapes: [{
      type: "line", x0: med, x1: med, y0: 0, y1: 1,
      xref: "x", yref: "paper",
      line: { color: "#1e293b", width: 1.5, dash: "dash" },
    }],
    annotations: [{
      x: med, y: 0.96, xref: "x", yref: "paper",
      text: `median = ${med.toFixed(1)}`, showarrow: false,
      font: { size: 8.5, color: "#1e293b", family: "Plus Jakarta Sans" },
      bgcolor: "rgba(255,255,255,.85)", borderpad: 2,
    }],
  }, { displayModeBar: false, responsive: true });
}
