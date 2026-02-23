/* ================================================================== *
 *  Stress Kt Estimator – frontend logic                              *
 * ================================================================== */

const COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"];
const KT_ORDER = [
    "Fx+","Fx-","Fy+","Fy-","Fz+","Fz-",
    "Mx+","Mx-","My+","My-","Mz+","Mz-",
];

let selectedRowIdx = -1;

/* ── Initialise ────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
    addRow();
    addRow();
    addRow();
    onSettingsChanged();
});

/* ================================================================== *
 *  Load-case table CRUD                                              *
 * ================================================================== */

function makeRow(vals) {
    const defaults = vals || { name: "", fx: 0, fy: 0, fz: 0, mx: 0, my: 0, mz: 0, stress: 0 };
    const tbody = document.getElementById("loadCaseBody");
    const tr = document.createElement("tr");
    const idx = tbody.rows.length;

    tr.innerHTML = `
        <td class="text-center text-muted" style="width:40px">${idx + 1}</td>
        <td><input type="text"   class="lc-name"   value="${esc(defaults.name)}"></td>
        <td><input type="number" class="lc-fx"     value="${defaults.fx}" step="any"></td>
        <td><input type="number" class="lc-fy"     value="${defaults.fy}" step="any"></td>
        <td><input type="number" class="lc-fz"     value="${defaults.fz}" step="any"></td>
        <td><input type="number" class="lc-mx"     value="${defaults.mx}" step="any"></td>
        <td><input type="number" class="lc-my"     value="${defaults.my}" step="any"></td>
        <td><input type="number" class="lc-mz"     value="${defaults.mz}" step="any"></td>
        <td><input type="number" class="lc-stress" value="${defaults.stress}" step="any"></td>`;

    tr.addEventListener("click", () => selectRow(tr));
    tr.querySelectorAll("input").forEach(inp =>
        inp.addEventListener("input", onSettingsChanged));
    tbody.appendChild(tr);
    return tr;
}

function addRow(vals) { makeRow(vals); onSettingsChanged(); }

function deleteRow() {
    const tbody = document.getElementById("loadCaseBody");
    if (selectedRowIdx >= 0 && selectedRowIdx < tbody.rows.length) {
        tbody.deleteRow(selectedRowIdx);
        selectedRowIdx = -1;
        renumberRows();
        onSettingsChanged();
    }
}

function selectRow(tr) {
    document.querySelectorAll("#loadCaseBody tr").forEach(r => r.classList.remove("selected"));
    tr.classList.add("selected");
    selectedRowIdx = Array.from(tr.parentElement.children).indexOf(tr);
}

function renumberRows() {
    document.querySelectorAll("#loadCaseBody tr").forEach((tr, i) => {
        tr.cells[0].textContent = i + 1;
    });
}

/* ── CSV import / export ──────────────────────────────────────────── */

function importCSV(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        const text = e.target.result.replace(/^\uFEFF/, "");   // strip BOM
        const lines = text.split(/\r?\n/).filter(l => l.trim());
        if (lines.length < 2) return alert("CSV must have a header row and at least one data row.");

        const hdr = lines[0].split(",").map(h => h.trim());
        const required = ["Case Name","Fx","Fy","Fz","Mx","My","Mz","Stress"];
        const missing = required.filter(r => !hdr.includes(r));
        if (missing.length) return alert("Missing columns: " + missing.join(", "));

        const ci = Object.fromEntries(required.map(c => [c, hdr.indexOf(c)]));

        document.getElementById("loadCaseBody").innerHTML = "";
        selectedRowIdx = -1;

        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(",").map(c => c.trim());
            if (cols.length < hdr.length) continue;
            addRow({
                name:   cols[ci["Case Name"]],
                fx:     parseFloat(cols[ci["Fx"]]) || 0,
                fy:     parseFloat(cols[ci["Fy"]]) || 0,
                fz:     parseFloat(cols[ci["Fz"]]) || 0,
                mx:     parseFloat(cols[ci["Mx"]]) || 0,
                my:     parseFloat(cols[ci["My"]]) || 0,
                mz:     parseFloat(cols[ci["Mz"]]) || 0,
                stress: parseFloat(cols[ci["Stress"]]) || 0,
            });
        }
        onSettingsChanged();
    };
    reader.readAsText(file);
    event.target.value = "";
}

function exportCSV() {
    const cases = collectLoadCases();
    if (!cases.length) return;
    let csv = "Case Name,Fx,Fy,Fz,Mx,My,Mz,Stress\n";
    cases.forEach(c => {
        csv += `${c.case_name},${c.fx},${c.fy},${c.fz},${c.mx},${c.my},${c.mz},${c.stress}\n`;
    });
    downloadText(csv, "load_cases.csv", "text/csv");
}

/* ================================================================== *
 *  Collect UI state                                                  *
 * ================================================================== */

function collectLoadCases() {
    const rows = document.querySelectorAll("#loadCaseBody tr");
    return Array.from(rows).map(tr => ({
        case_name: tr.querySelector(".lc-name").value || "LC",
        fx:     num(tr.querySelector(".lc-fx").value),
        fy:     num(tr.querySelector(".lc-fy").value),
        fz:     num(tr.querySelector(".lc-fz").value),
        mx:     num(tr.querySelector(".lc-mx").value),
        my:     num(tr.querySelector(".lc-my").value),
        mz:     num(tr.querySelector(".lc-mz").value),
        stress: num(tr.querySelector(".lc-stress").value),
    }));
}

function collectSettings() {
    const useSep = document.getElementById("useSeparateSign").checked;
    const modes = [];
    const fixedKt = [];
    document.querySelectorAll(".comp-mode").forEach(sel => {
        modes.push(sel.value);
        const row = sel.closest(".component-row");
        fixedKt.push([
            num(row.querySelector(".kt-plus")?.value),
            num(row.querySelector(".kt-minus")?.value),
        ]);
    });
    return {
        use_separate_sign: useSep,
        sign_mode_per_component: useSep ? modes : null,
        fixed_kt_values: useSep ? fixedKt : null,
        safety_factor: num(document.getElementById("safetyFactor").value) || 1.0,
    };
}

function collectKtValues() {
    const inputs = document.querySelectorAll("#ktValues input");
    return Array.from(inputs).map(inp => num(inp.value));
}

/* ================================================================== *
 *  Constraint status                                                 *
 * ================================================================== */

function onSettingsChanged() {
    updateConstraintStatus();
}

function onModeChanged(sel) {
    const row = sel.closest(".component-row");
    const manualDiv = row.querySelector(".manual-kt");
    manualDiv.classList.toggle("d-none", sel.value !== "set");
    onSettingsChanged();
}

function updateConstraintStatus() {
    const nCases = document.querySelectorAll("#loadCaseBody tr").length;
    const useSep = document.getElementById("useSeparateSign").checked;
    let nVars = 0;
    if (!useSep) {
        nVars = 6;
    } else {
        document.querySelectorAll(".comp-mode").forEach(sel => {
            if (sel.value === "linked") nVars += 1;
            else if (sel.value === "individual") nVars += 2;
        });
    }

    const badge = document.getElementById("constraintStatus");
    const label = document.getElementById("constraintLabel");
    const detail = document.getElementById("constraintDetail");

    badge.className = "constraint-badge mb-3";
    if (nCases === 0) {
        badge.classList.add("status-none");
        label.textContent = "No load cases";
        detail.textContent = "";
    } else if (nCases < nVars) {
        badge.classList.add("status-red");
        label.textContent = "Under-constrained";
        detail.textContent = `${nCases} cases < ${nVars} variables`;
    } else if (nCases === nVars) {
        badge.classList.add("status-orange");
        label.textContent = "Just determined";
        detail.textContent = `${nCases} cases = ${nVars} variables`;
    } else {
        badge.classList.add("status-green");
        label.textContent = "Well constrained";
        detail.textContent = `${nCases} cases > ${nVars} variables`;
    }
}

/* ================================================================== *
 *  API calls                                                         *
 * ================================================================== */

async function doSolve() {
    const cases = collectLoadCases();
    if (!cases.length) return alert("Add at least one load case.");
    setLoading("btnSolve", true);
    try {
        const data = await apiPost("/api/solve", {
            load_cases: cases,
            settings: collectSettings(),
        });
        displayResult(data);
    } catch (e) {
        showError(e);
    } finally {
        setLoading("btnSolve", false);
    }
}

async function doRecalc() {
    const cases = collectLoadCases();
    const kt = collectKtValues();
    if (!cases.length) return alert("Add at least one load case.");
    if (kt.length !== 12) return alert("Solve first to populate Kt values.");
    setLoading("btnRecalc", true);
    try {
        const data = await apiPost("/api/recalc", {
            load_cases: cases,
            settings: collectSettings(),
            kt_values: kt,
        });
        displayResult(data);
    } catch (e) {
        showError(e);
    } finally {
        setLoading("btnRecalc", false);
    }
}

async function suggestUnlink() {
    const cases = collectLoadCases();
    if (cases.length < 2) return alert("Need at least 2 load cases.");
    setLoading("btnSuggest", true);
    try {
        const data = await apiPost("/api/suggest-unlink", { load_cases: cases });
        const suggested = data.suggested_components || [];
        if (!suggested.length) {
            alert("No components suggested for unlinking.");
            return;
        }
        document.getElementById("useSeparateSign").checked = true;
        document.querySelectorAll(".comp-mode").forEach(sel => {
            sel.value = suggested.includes(sel.dataset.comp) ? "individual" : "linked";
            onModeChanged(sel);
        });
        alert("Suggested: set " + suggested.join(", ") + " to Individual.");
    } catch (e) {
        showError(e);
    } finally {
        setLoading("btnSuggest", false);
    }
}

async function findMinimalUnlink() {
    const cases = collectLoadCases();
    if (!cases.length) return alert("Add at least one load case.");
    setLoading("btnFindMinimal", true);
    try {
        const data = await apiPost("/api/find-minimal-unlink", {
            load_cases: cases,
            settings: collectSettings(),
        });
        if (data.sign_modes) {
            document.getElementById("useSeparateSign").checked = true;
            const sels = document.querySelectorAll(".comp-mode");
            data.sign_modes.forEach((m, i) => {
                if (sels[i]) { sels[i].value = m; onModeChanged(sels[i]); }
            });
        }
        if (data.result) displayResult(data.result);
    } catch (e) {
        showError(e);
    } finally {
        setLoading("btnFindMinimal", false);
    }
}

/* ================================================================== *
 *  Display results                                                   *
 * ================================================================== */

function displayResult(r) {
    const statusBadge = document.getElementById("resultStatus");
    statusBadge.className = "badge " + (r.success ? "bg-success" : "bg-danger");
    statusBadge.textContent = r.success ? "Success" : "Failed";

    // Summary
    const summaryDiv = document.getElementById("resultSummary");
    const cards = document.getElementById("summaryCards");
    summaryDiv.classList.remove("d-none");
    cards.innerHTML = [
        metric("Worst margin", fmt(r.worst_case_margin, 2) + "%"),
        metric("Max overpredict.", fmt(r.max_overprediction, 3)),
        metric("Max underpredict.", fmt(r.max_underprediction, 3)),
        metric("RMS error", fmt(r.rms_error, 3)),
        metric("Condition #", fmt(r.condition_number, 1)),
    ].join("");

    if (r.diagnostics) {
        const d = r.diagnostics;
        if (d.constraint_status && d.constraint_status !== "well_determined" && d.constraint_status !== "recalc_fixed_kt") {
            summaryDiv.classList.remove("alert-light");
            summaryDiv.classList.add("alert-warning");
        } else {
            summaryDiv.classList.remove("alert-warning");
            summaryDiv.classList.add("alert-light");
        }
    }

    // Kt table
    const ktSection = document.getElementById("ktSection");
    const ktHeader = document.getElementById("ktHeader");
    const ktRow = document.getElementById("ktValues");
    ktSection.classList.remove("d-none");
    ktHeader.innerHTML = r.kt_names.map(n => `<th>${esc(n)}</th>`).join("");
    ktRow.innerHTML = r.kt_values.map(v =>
        `<td><input type="number" value="${fmt(v, 6)}" step="any"></td>`
    ).join("");

    // Per-case table
    const pcSection = document.getElementById("perCaseSection");
    const pcBody = document.getElementById("perCaseBody");
    pcSection.classList.remove("d-none");
    pcBody.innerHTML = r.per_case.map(c => {
        const cls = c.margin_pct < 0 ? "text-danger fw-bold" : "";
        return `<tr>
            <td>${esc(c.case_name)}</td>
            <td>${fmt(c.actual, 3)}</td>
            <td>${fmt(c.predicted, 3)}</td>
            <td class="${cls}">${fmt(c.margin_pct, 2)}%</td>
        </tr>`;
    }).join("");

    // Fetch plots
    fetchPlots(r);
}

async function fetchPlots(r) {
    const plotSection = document.getElementById("plotSection");
    try {
        const data = await apiPost("/api/plot", {
            kt_names: r.kt_names,
            kt_values: r.kt_values,
            sigma_target: r.sigma_target,
            sigma_pred: r.sigma_pred,
            per_case: r.per_case,
        });
        document.getElementById("plotBar").src = "data:image/png;base64," + data.bar_chart;
        document.getElementById("plotScatter").src = "data:image/png;base64," + data.scatter_chart;
        plotSection.classList.remove("d-none");
    } catch (e) {
        console.error("Plot error:", e);
    }
}

/* ================================================================== *
 *  Utilities                                                         *
 * ================================================================== */

async function apiPost(url, body) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.disabled = loading;
    if (loading) {
        btn.dataset.origHtml = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Working…';
    } else {
        btn.innerHTML = btn.dataset.origHtml || btn.innerHTML;
    }
}

function showError(e) {
    alert("Error: " + (e.message || e));
}

function metric(label, value) {
    return `<div class="col summary-metric">
        <div class="value">${value}</div>
        <div class="label">${label}</div>
    </div>`;
}

function downloadText(text, filename, mime) {
    const blob = new Blob([text], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

function num(v) { const n = parseFloat(v); return isNaN(n) ? 0 : n; }
function fmt(v, d) { return (typeof v === "number" && isFinite(v)) ? v.toFixed(d) : "—"; }
function esc(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
}
