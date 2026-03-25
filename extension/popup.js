/**
 * YouTube AI Media Detector – Popup Script  v1.1.0
 *
 * Pings the local FastAPI backend on popup open to display:
 * - Online / Offline status
 * - Which AI models are loaded
 * - Active compute device (CPU / CUDA)
 */

const BACKEND_URL = "http://localhost:8000";

async function checkBackend() {
    const pill = document.getElementById("status-pill");
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    const modelsSection = document.getElementById("models-section");
    const deviceRow = document.getElementById("device-row");
    const deviceVal = document.getElementById("device-val");

    try {
        const resp = await fetch(`${BACKEND_URL}/status`, { signal: AbortSignal.timeout(4000) });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const data = await resp.json();

        // ── Online ──
        pill.className = "status-pill online";
        dot.className = "status-dot online";
        text.textContent = "Online";

        // ── Model states ──
        modelsSection.style.display = "block";
        deviceRow.style.display = "flex";

        const models = data.models || {};
        document.getElementById("model-clip").innerHTML = models.clip ? "✅" : "⚠️";
        document.getElementById("model-mtcnn").innerHTML = models.mtcnn ? "✅" : "⚠️";
        document.getElementById("model-deepfake").innerHTML = models.deepfake_vit ? "✅" : "⚠️";

        // ── Compute device ──
        const deviceRaw = (data.device || "cpu").toUpperCase();
        const hasCuda = data.cuda_available;
        deviceVal.textContent = hasCuda ? `${deviceRaw}  🚀` : deviceRaw;

    } catch (_) {
        // ── Offline ──
        pill.className = "status-pill offline";
        dot.className = "status-dot offline";
        text.textContent = "Offline";
        modelsSection.style.display = "none";
        deviceRow.style.display = "none";
    }
}

// Run immediately when popup opens
checkBackend();
