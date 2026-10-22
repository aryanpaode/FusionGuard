let activeStream = null;
let activeMode = null;
let monitoring = false;
let monitorBusy = false;
let monitorToken = 0;
let cooldownTimer = null;
let stableReadyCount = 0;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function setStatus(message) {
    const el = document.getElementById("monitor-status");
    if (el) el.textContent = message;
}

function stopActiveCamera() {
    if (activeStream) activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
    activeMode = null;
    ["monitor-camera", "enrol-camera"].forEach(id => {
        const video = document.getElementById(id);
        if (video) {
            video.pause();
            video.srcObject = null;
        }
    });
}

async function startCamera(mode) {
    const video = document.getElementById(mode === "monitor" ? "monitor-camera" : "enrol-camera");
    if (!video) throw new Error("Camera preview is unavailable.");

    if (activeStream && activeMode === mode && video.srcObject === activeStream) return activeStream;
    stopActiveCamera();

    activeStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280, min: 640 }, height: { ideal: 720, min: 480 }, facingMode: "user" },
        audio: false
    });
    activeMode = mode;
    video.srcObject = activeStream;
    await video.play();
    return activeStream;
}

function captureFrame(videoId, canvasId, jpegQuality = 0.92) {
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    if (!activeStream || !video || video.srcObject !== activeStream) throw new Error("Start the correct webcam first.");
    if (!video.videoWidth || !video.videoHeight) throw new Error("Wait for the webcam preview to load.");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", jpegQuality);
}

async function startEnrolCamera() {
    monitoring = false;
    monitorToken += 1;
    if (cooldownTimer) clearInterval(cooldownTimer);
    try {
        await startCamera("enrol");
    } catch (error) {
        alert("The enrolment webcam could not be opened: " + error.message);
    }
}

function captureEnrolment(angleKey) {
    try {
        const dataUrl = captureFrame("enrol-camera", "enrol-canvas", 0.88);
        document.getElementById(angleKey + "_image").value = dataUrl;
        document.getElementById(angleKey + "-preview").src = dataUrl;
        document.getElementById(angleKey + "-card").classList.add("captured");
    } catch (error) {
        alert(error.message);
    }
}

const enrolForm = document.getElementById("enrol-form");
if (enrolForm) {
    enrolForm.addEventListener("submit", event => {
        const missing = ["front", "left", "right", "up"].filter(key => !document.getElementById(key + "_image").value);
        if (missing.length) {
            event.preventDefault();
            alert("Capture all four views before enrolment: " + missing.join(", "));
            return;
        }
        stopActiveCamera();
    });
}

async function startMonitoring() {
    monitoring = true;
    monitorBusy = false;
    stableReadyCount = 0;
    monitorToken += 1;
    const token = monitorToken;
    document.getElementById("resume-button").disabled = true;
    try {
        await startCamera("monitor");
        setStatus("Monitoring active. Keep one face visible and steady.");
        probeLoop(token);
    } catch (error) {
        monitoring = false;
        setStatus("Camera error: " + error.message);
        alert("The recognition webcam could not be opened: " + error.message);
    }
}

function stopMonitoring() {
    monitoring = false;
    monitorBusy = false;
    stableReadyCount = 0;
    monitorToken += 1;
    if (cooldownTimer) clearInterval(cooldownTimer);
    cooldownTimer = null;
    document.getElementById("resume-button").disabled = true;
    stopActiveCamera();
    setStatus("Monitoring stopped.");
}

async function postJson(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {})
    });
    let data = {};
    try { data = await response.json(); } catch (_) { data = { ok: false, error: "Server returned a non-JSON response." }; }
    data.httpStatus = response.status;
    return data;
}

async function probeLoop(token) {
    while (monitoring && token === monitorToken) {
        if (monitorBusy) {
            await sleep(250);
            continue;
        }

        try {
            const image = captureFrame("monitor-camera", "monitor-canvas", 0.94);
            const data = await postJson("/api/probe", { image });
            if (!monitoring || token !== monitorToken) return;

            if (data.ok && data.ready) {
                stableReadyCount += 1;
                const warning = data.quality?.warning ? ` (${data.quality.warning})` : "";
                setStatus(`Face ready ${stableReadyCount}/${window.FG_CONFIG.stableProbeCount}${warning}. Keep still.`);
                if (stableReadyCount >= window.FG_CONFIG.stableProbeCount) {
                    monitorBusy = true;
                    stableReadyCount = 0;
                    await runRecognitionSession(token);
                }
            } else {
                stableReadyCount = 0;
                const reason = data.quality?.reason || data.error || "Waiting for a clear face";
                setStatus("Waiting: " + reason + ".");
            }
        } catch (error) {
            stableReadyCount = 0;
            setStatus("Probe error: " + error.message);
        }
        await sleep(900);
    }
}

async function captureBurst(count, intervalMs, messagePrefix) {
    const frames = [];
    for (let i = 0; i < count; i += 1) {
        if (!monitoring) break;
        setStatus(`${messagePrefix} ${i + 1}/${count}`);
        frames.push(captureFrame("monitor-camera", "monitor-canvas", 0.92));
        if (i < count - 1) await sleep(intervalMs);
    }
    return frames;
}

async function runRecognitionSession(token) {
    try {
        setStatus("Liveness check: blink once or turn your head slightly.");
        await sleep(350);
        const livenessFrames = await captureBurst(window.FG_CONFIG.livenessFrames, 330, "Collecting separate liveness frame");
        if (!monitoring || token !== monitorToken) return;

        setStatus("Liveness captured. Hold your face naturally for recognition.");
        await sleep(450);
        const recognitionFrames = await captureBurst(window.FG_CONFIG.recognitionFrames, 280, "Collecting recognition frame");
        if (!monitoring || token !== monitorToken) return;

        setStatus("Running V3 multimodal recognition and temporal voting...");
        const result = await postJson("/api/monitor/recognize", {
            liveness_frames: livenessFrames,
            recognition_frames: recognitionFrames
        });
        if (!monitoring || token !== monitorToken) return;

        if (result.ok) {
            showResult(result);
            if (result.alert) playAlertTone();
            startCooldown(result.cooldown_seconds || window.FG_CONFIG.cooldownSeconds, token);
            return;
        }

        if (result.cooldown) {
            startCooldown(result.remaining || window.FG_CONFIG.cooldownSeconds, token);
            return;
        }

        if (result.retry) {
            setStatus(result.error + " Reposition and keep looking toward the camera.");
            monitorBusy = false;
            await sleep(1000);
            return;
        }

        setStatus("Recognition error: " + (result.error || "Unknown error"));
        monitorBusy = false;
    } catch (error) {
        setStatus("Recognition error: " + error.message);
        monitorBusy = false;
    }
}

function startCooldown(seconds, token) {
    monitorBusy = true;
    document.getElementById("resume-button").disabled = false;
    let remaining = seconds;
    setStatus(`Recognition complete. Cooldown ${remaining}s — preview remains active.`);
    if (cooldownTimer) clearInterval(cooldownTimer);
    cooldownTimer = setInterval(() => {
        remaining -= 1;
        if (!monitoring || token !== monitorToken) {
            clearInterval(cooldownTimer);
            cooldownTimer = null;
            return;
        }
        if (remaining <= 0) {
            clearInterval(cooldownTimer);
            cooldownTimer = null;
            monitorBusy = false;
            stableReadyCount = 0;
            document.getElementById("resume-button").disabled = true;
            setStatus("Cooldown complete. Monitoring for the next stable face.");
        } else {
            setStatus(`Recognition complete. Cooldown ${remaining}s — no new recognition or logs.`);
        }
    }, 1000);
}

async function resumeNow() {
    if (!monitoring) return;
    const result = await postJson("/api/cooldown/resume", {});
    if (!result.ok) {
        setStatus("Could not resume: " + (result.error || "server error"));
        return;
    }
    if (cooldownTimer) clearInterval(cooldownTimer);
    cooldownTimer = null;
    monitorBusy = false;
    stableReadyCount = 0;
    document.getElementById("resume-button").disabled = true;
    setStatus("Cooldown cleared. Monitoring resumed now.");
}

function showResult(result) {
    const panel = document.getElementById("recognition-result");
    panel.className = "result-panel show";
    if (result.decision === "Authorised") panel.classList.add("authorised");
    else if (result.decision === "Uncertain") panel.classList.add("uncertain");
    else if (result.decision === "Suspected Spoof") panel.classList.add("spoof");
    else panel.classList.add("unknown");

    let title = result.decision;
    if (result.decision === "Authorised" && result.winner_name) title += ": " + result.winner_name;
    document.getElementById("result-title").textContent = title;

    const liveText = result.liveness?.passed ? "Liveness passed" : "Liveness failed";
    const incidentText = result.incident_id ? ` • Incident #${result.incident_id}` : "";
    document.getElementById("result-summary").textContent = `${liveText} • Session #${result.session_id}${incidentText}`;

    document.getElementById("r-arcface").textContent = `${result.arcface_similarity}%`;
    document.getElementById("r-geometry").textContent = `${result.geometry_similarity}%`;
    document.getElementById("r-mahal").textContent = String(result.mahalanobis_distance);
    document.getElementById("r-iris").textContent = `${result.iris_similarity}%`;
    document.getElementById("r-fusion").textContent = `${result.median_fusion_score}%`;
    document.getElementById("r-confidence").textContent = `${(result.median_confidence * 100).toFixed(2)}%`;
    document.getElementById("r-votes").textContent = `${result.winner_votes}/${result.required_votes} of ${result.usable_frames}`;
    document.getElementById("r-angle").textContent = result.best_angle_label || "—";
    refreshIncidents(true);
}

function playAlertTone() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioContext();
        [0, 0.28, 0.56].forEach(delay => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "square";
            osc.frequency.value = 760;
            gain.gain.setValueAtTime(0.0001, ctx.currentTime + delay);
            gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + delay + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + delay + 0.18);
            osc.connect(gain).connect(ctx.destination);
            osc.start(ctx.currentTime + delay);
            osc.stop(ctx.currentTime + delay + 0.2);
        });
    } catch (_) {
        // Audio is a best-effort browser alert; server-side incident logging is authoritative.
    }
}

window.addEventListener("beforeunload", stopActiveCamera);


// ---------------------------------------------------------------------------
// V3.4 dynamic incident log and evidence viewer
// ---------------------------------------------------------------------------
let currentIncidentId = null;
let incidentRefreshTimer = null;

function textOrDash(value) {
    return value === null || value === undefined || value === "" ? "—" : String(value);
}

function percent(value, scale = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${(number * scale).toFixed(2)}%`;
}

function median(values) {
    const numbers = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!numbers.length) return null;
    const middle = Math.floor(numbers.length / 2);
    return numbers.length % 2 ? numbers[middle] : (numbers[middle - 1] + numbers[middle]) / 2;
}

function makeCell(row, text) {
    const td = document.createElement("td");
    td.textContent = text;
    row.appendChild(td);
    return td;
}

function pill(text, className) {
    const span = document.createElement("span");
    span.className = `pill ${className}`;
    span.textContent = text;
    return span;
}

function updateIncidentDeleteButton() {
    const checked = document.querySelectorAll(".incident-select:checked").length;
    const button = document.getElementById("delete-incidents-button");
    if (button) {
        button.disabled = checked === 0;
        button.textContent = checked ? `Delete Selected Logs (${checked})` : "Delete Selected Logs";
    }
    const all = document.querySelectorAll(".incident-select");
    const selectAll = document.getElementById("incident-select-all");
    if (selectAll) {
        selectAll.checked = all.length > 0 && checked === all.length;
        selectAll.indeterminate = checked > 0 && checked < all.length;
    }
}

function renderIncidents(items) {
    const tbody = document.getElementById("incident-table-body");
    if (!tbody) return;
    const selected = new Set([...document.querySelectorAll(".incident-select:checked")].map(el => String(el.value)));
    tbody.replaceChildren();

    if (!items.length) {
        const row = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 10;
        td.textContent = "No incidents recorded for this filter.";
        row.appendChild(td);
        tbody.appendChild(row);
        updateIncidentDeleteButton();
        return;
    }

    items.forEach(item => {
        const row = document.createElement("tr");
        const selectTd = document.createElement("td");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "incident-select";
        checkbox.value = item.id;
        checkbox.checked = selected.has(String(item.id));
        checkbox.setAttribute("aria-label", `Select incident ${item.id}`);
        checkbox.addEventListener("change", updateIncidentDeleteButton);
        selectTd.appendChild(checkbox);
        row.appendChild(selectTd);

        makeCell(row, `#${item.id}`);
        makeCell(row, textOrDash(item.created_at));
        makeCell(row, textOrDash(item.session_decision || "Guest / Other"));
        makeCell(row, textOrDash(item.best_candidate_name));
        const votes = item.session_id ? `${item.winning_votes ?? 0}/${item.required_votes ?? 0} of ${item.usable_frames ?? 0}` : "—";
        makeCell(row, votes);

        const severityTd = document.createElement("td");
        severityTd.appendChild(pill(textOrDash(item.severity), item.severity === "high" ? "red" : "amber"));
        row.appendChild(severityTd);
        makeCell(row, textOrDash(item.reason));

        const statusTd = document.createElement("td");
        statusTd.appendChild(pill(textOrDash(item.status), "gray"));
        row.appendChild(statusTd);

        const detailsTd = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "small";
        button.textContent = "View Details";
        button.addEventListener("click", () => openIncidentDetails(item.id));
        detailsTd.appendChild(button);
        row.appendChild(detailsTd);
        tbody.appendChild(row);
    });
    updateIncidentDeleteButton();
}

async function refreshIncidents(quiet = false) {
    const tbody = document.getElementById("incident-table-body");
    if (!tbody) return;
    const status = document.getElementById("incident_status")?.value || "";
    const state = document.getElementById("incident-refresh-state");
    try {
        const response = await fetch(`/api/incidents?status=${encodeURIComponent(status)}`, { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Could not refresh incidents.");
        renderIncidents(data.incidents || []);
        const count = document.getElementById("open-incident-count");
        if (count) count.textContent = data.open_incidents;
        if (state) {
            state.textContent = "Auto-refresh: live";
            state.classList.add("live");
        }
    } catch (error) {
        if (state) {
            state.textContent = "Auto-refresh paused";
            state.classList.remove("live");
        }
        if (!quiet) console.error(error);
    }
}

function detailBox(label, value) {
    const div = document.createElement("div");
    div.className = "detail-box";
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = textOrDash(value);
    div.append(span, strong);
    return div;
}

function buildFrameTable(frames, identityModel = "ArcFace") {
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    const identityLabel = String(identityModel || "").toLowerCase() === "arcface" ? "ArcFace" : "CNN";
    table.innerHTML = `<thead><tr><th>Frame</th><th>Candidate</th><th>Best Angle</th><th>Frame Decision</th><th>${identityLabel}</th><th>Geometry</th><th>MD</th><th>Iris/LBP</th><th>Fusion</th><th>Confidence</th><th>Margin</th><th>Reason</th></tr></thead>`;
    const tbody = document.createElement("tbody");
    frames.forEach(frame => {
        const row = document.createElement("tr");
        makeCell(row, String(frame.frame_index));
        makeCell(row, textOrDash(frame.candidate_name));
        makeCell(row, textOrDash(frame.best_angle_label));
        makeCell(row, textOrDash(frame.frame_decision));
        makeCell(row, percent(frame.cnn_similarity));
        makeCell(row, percent(frame.geometry_similarity));
        makeCell(row, Number(frame.mahalanobis_distance).toFixed(3));
        makeCell(row, percent(frame.iris_similarity));
        makeCell(row, percent(frame.fusion_score));
        makeCell(row, percent(frame.logistic_probability, 100));
        makeCell(row, frame.identity_margin === null || frame.identity_margin === undefined ? "—" : Number(frame.identity_margin).toFixed(3));
        makeCell(row, textOrDash(frame.rejection_reason));
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
}

async function openIncidentDetails(incidentId) {
    const modal = document.getElementById("incident-modal");
    const body = document.getElementById("incident-detail-body");
    if (!modal || !body) return;
    currentIncidentId = Number(incidentId);
    modal.hidden = false;
    document.body.style.overflow = "hidden";
    body.innerHTML = '<p class="subtext">Loading incident evidence…</p>';

    try {
        const response = await fetch(`/api/incidents/${incidentId}`, { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Incident could not be loaded.");
        const incident = data.incident;
        const frames = data.frames || [];
        const alerts = data.alerts || [];

        document.getElementById("incident-modal-title").textContent = `Incident #${incident.id}`;
        document.getElementById("incident-modal-subtitle").textContent = `${textOrDash(incident.created_at)} • ${textOrDash(incident.severity)} severity`;
        document.getElementById("detail-incident-status").value = incident.status || "open";
        document.getElementById("detail-incident-notes").value = incident.notes || "";

        body.replaceChildren();
        const reason = document.createElement("div");
        reason.className = "detail-reason";
        reason.textContent = incident.reason || "No reason stored.";
        body.appendChild(reason);

        const grid = document.createElement("div");
        grid.className = "detail-grid";
        grid.append(
            detailBox("Recognition session", incident.session_id ? `#${incident.session_id}` : "Not linked"),
            detailBox("Decision", incident.session_decision || "Guest / Other"),
            detailBox("Nearest candidate", incident.best_candidate_name || "—"),
            detailBox("Liveness", incident.liveness_result || "—"),
            detailBox("Votes", incident.session_id ? `${incident.winning_votes ?? 0}/${incident.required_votes ?? 0} of ${incident.usable_frames ?? 0}` : "—"),
            detailBox("Median Fusion", incident.session_id ? percent(incident.median_fusion_score) : "—"),
            detailBox("Median Confidence", incident.session_id ? percent(incident.median_confidence, 100) : "—"),
            detailBox("Identity Margin", incident.session_id ? textOrDash(incident.median_identity_margin) : "—")
        );
        body.appendChild(grid);

        if (frames.length) {
            const componentGrid = document.createElement("div");
            componentGrid.className = "detail-grid";
            componentGrid.append(
                detailBox((incident.identity_model || "").toLowerCase() === "arcface" ? "Median ArcFace" : "Median CNN", percent(median(frames.map(f => f.cnn_similarity)))),
                detailBox("Median Geometry", percent(median(frames.map(f => f.geometry_similarity)))),
                detailBox("Median Mahalanobis", median(frames.map(f => f.mahalanobis_distance))?.toFixed(3) ?? "—"),
                detailBox("Median Iris/LBP", percent(median(frames.map(f => f.iris_similarity))))
            );
            body.appendChild(componentGrid);
        }

        if (incident.evidence_url) {
            const evidence = document.createElement("p");
            const link = document.createElement("a");
            link.href = incident.evidence_url;
            link.target = "_blank";
            link.rel = "noopener";
            link.textContent = "Open captured evidence image";
            evidence.appendChild(link);
            body.appendChild(evidence);
        }

        const alertTitle = document.createElement("h3");
        alertTitle.className = "detail-section-title";
        alertTitle.textContent = `Alerts (${alerts.length})`;
        body.appendChild(alertTitle);
        if (alerts.length) {
            alerts.forEach(alertItem => {
                const item = document.createElement("div");
                item.className = "alert-item";
                item.textContent = `${textOrDash(alertItem.created_at)} • ${textOrDash(alertItem.alert_type)} • ${textOrDash(alertItem.message)}`;
                body.appendChild(item);
            });
        } else {
            const empty = document.createElement("p");
            empty.className = "subtext";
            empty.textContent = "No alert rows are linked to this incident.";
            body.appendChild(empty);
        }

        const frameTitle = document.createElement("h3");
        frameTitle.className = "detail-section-title";
        frameTitle.textContent = `Frame-by-frame recognition evidence (${frames.length})`;
        body.appendChild(frameTitle);
        if (frames.length) body.appendChild(buildFrameTable(frames, incident.identity_model));
        else {
            const empty = document.createElement("p");
            empty.className = "subtext";
            empty.textContent = incident.session_id ? "No frame-result rows were stored for this session." : "This incident was not created by a biometric recognition session.";
            body.appendChild(empty);
        }
    } catch (error) {
        body.innerHTML = "";
        const msg = document.createElement("p");
        msg.className = "warning";
        msg.textContent = error.message;
        body.appendChild(msg);
    }
}

function closeIncidentDetails() {
    const modal = document.getElementById("incident-modal");
    if (modal) modal.hidden = true;
    currentIncidentId = null;
    document.body.style.overflow = "";
}

async function saveIncidentReview() {
    if (!currentIncidentId) return;
    const status = document.getElementById("detail-incident-status").value;
    const notes = document.getElementById("detail-incident-notes").value;
    const data = await postJson(`/api/incidents/${currentIncidentId}/review`, { status, notes });
    if (!data.ok) {
        alert(data.error || "Incident classification could not be saved.");
        return;
    }
    await refreshIncidents();
    await openIncidentDetails(currentIncidentId);
}

async function deleteSelectedIncidents() {
    const ids = [...document.querySelectorAll(".incident-select:checked")].map(el => Number(el.value)).filter(Number.isInteger);
    if (!ids.length) return;
    const message = `Delete ${ids.length} selected incident log${ids.length === 1 ? "" : "s"}?\n\nThis removes the incident row and its alert log(s). The underlying recognition-session record and captured biometric evidence are retained.`;
    if (!window.confirm(message)) return;
    const data = await postJson("/api/incidents/delete", { incident_ids: ids });
    if (!data.ok) {
        alert(data.error || "Selected incident logs could not be deleted.");
        return;
    }
    closeIncidentDetails();
    await refreshIncidents();
}

const incidentStatus = document.getElementById("incident_status");
if (incidentStatus) incidentStatus.addEventListener("change", () => refreshIncidents());

const incidentSelectAll = document.getElementById("incident-select-all");
if (incidentSelectAll) {
    incidentSelectAll.addEventListener("change", () => {
        document.querySelectorAll(".incident-select").forEach(box => { box.checked = incidentSelectAll.checked; });
        updateIncidentDeleteButton();
    });
}

document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeIncidentDetails();
});

const incidentModal = document.getElementById("incident-modal");
if (incidentModal) {
    incidentModal.addEventListener("click", event => {
        if (event.target === incidentModal) closeIncidentDetails();
    });
}

if (document.getElementById("incident-table-body")) {
    refreshIncidents(true);
    incidentRefreshTimer = setInterval(() => refreshIncidents(true), 3000);
}
