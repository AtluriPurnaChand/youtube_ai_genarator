/**
 * YouTube AI Media Detector – Content Script  v1.3.0
 *
 * ADAPTIVE PARALLEL PROGRESSIVE SAMPLING
 * ──────────────────────────────────────
 *  Frame count and interval scale automatically with video.duration:
 *
 *  Duration          Frames   Interval   Sample window
 *  ──────────────────────────────────────────────────
 *  < 30 s  (Short)      4       5 s        ~20 s
 *  30–90 s (Short)      6       8 s        ~45 s
 *  1.5–5 min           12      12 s       ~2.2 min
 *  5–15 min            16      18 s       ~4.8 min
 *  > 15 min            20      25 s        ~8 min
 *
 *  All requests are fired IN PARALLEL — badge updates LIVE as each
 *  response arrives so you see a result within seconds, not minutes.
 */

; (function () {
  "use strict";

  /* ================================================================
     CONFIG  (bounds only — actual counts computed per video)
  ================================================================ */
  const API_URL = "http://localhost:8000/analyze";
  const BADGE_ID = "yt-ai-detector-badge";
  const RECYCLE_WAIT = 60000;   // ms before re-scanning after a cycle ends

  const MIN_FRAMES = 4;       // never fewer than 4 frames
  const MAX_FRAMES = 20;      // never more than 20 frames
  const MIN_INTERVAL = 3000;    // at least 3 s between captures
  const MAX_INTERVAL = 30000;   // at most 30 s between captures
  const COVERAGE = 0.85;    // cover this fraction of video duration
  const MAX_WINDOW = 10 * 60 * 1000;  // cap sampling window at 10 min

  /**
   * Compute frame count & interval from video duration.
   *
   *  Duration  → Frames  Interval   Window
   *  < 30 s        4       5 s      ~20 s
   *  30–90 s       6       8 s      ~45 s
   *  1.5–5 min    12      12 s     ~2 min
   *  5–15 min     16      18 s     ~5 min
   *  > 15 min     20      25 s     ~8 min
   */
  function computeSamplingPlan(durationSec) {
    // 1 frame per ~45 s of video (rounded), clamped to [MIN_FRAMES, MAX_FRAMES]
    const rawFrames = Math.round(durationSec / 45);
    const frameCount = Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, rawFrames));

    // Sample window = coverage fraction of video, capped at MAX_WINDOW
    const windowMs = Math.min(durationSec * COVERAGE * 1000, MAX_WINDOW);

    // Spread frameCount captures evenly across windowMs
    const rawInterval = windowMs / Math.max(frameCount - 1, 1);
    const intervalMs = Math.min(MAX_INTERVAL, Math.max(MIN_INTERVAL, Math.round(rawInterval)));

    return { frameCount, intervalMs, windowMs };
  }

  /* ================================================================
     LABEL MAP
  ================================================================ */
  const LABELS = {
    real_video: { text: "Real Video", icon: "✅", cls: "badge-real" },
    ai_generated: { text: "AI Generated", icon: "🤖", cls: "badge-ai" },
    cartoon_animation: { text: "Cartoon / Anime", icon: "🎨", cls: "badge-cartoon" },
    video_game: { text: "Video Game", icon: "🎮", cls: "badge-game" },
    deepfake_detected: { text: "⚠ Deepfake", icon: "👤", cls: "badge-deepfake" },
    error: { text: "Detection Error", icon: "❓", cls: "badge-error" },
    offline: { text: "Backend Offline", icon: "🔌", cls: "badge-offline" },
  };

  /* ================================================================
     STATE
  ================================================================ */
  let cycleTimer = null;
  let lastVideoEl = null;
  let lastPlayerEl = null;
  let isRunning = false;

  let accumulatedResults = [];
  let framesCaptured = 0;

  /* ================================================================
     UTILITIES
  ================================================================ */

  function findVideo() {
    const candidates = Array.from(document.querySelectorAll("video"));
    return candidates
      .filter(v => v.readyState >= 2 && v.videoWidth > 0)
      .sort((a, b) => (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight))[0]
      || candidates[0]
      || null;
  }

  function findPlayerContainer(videoEl) {
    let el = videoEl?.parentElement;
    while (el) {
      if (el.id === "movie_player" || el.classList.contains("html5-video-player")) return el;
      if (
        el.tagName === "YTD-SHORTS" ||
        el.classList.contains("reel-video-in-sequence") ||
        el.id === "shorts-container" ||
        el.classList.contains("video-stream")
      ) return el;
      el = el.parentElement;
    }
    return videoEl?.parentElement || document.body;
  }

  /**
   * Confidence-weighted majority vote.
   * Deepfake safety-first: any high-confidence deepfake detection overrides.
   */
  function pickBestResult(results) {
    if (!results.length) return null;

    const valid = results.filter(r => r.type !== "error" && r.type !== "offline");
    if (!valid.length) {
      if (results.some(r => r.detail === "Backend Offline")) return { type: "offline", confidence: 0 };
      return { type: "error", confidence: 0, detail: results[0]?.detail || "Backend error" };
    }

    // Deepfake priority
    const deepfakes = valid.filter(r => r.type === "deepfake_detected");
    if (deepfakes.length) {
      const highConf = deepfakes.find(r => r.confidence >= 0.72);
      const suspicious = deepfakes.filter(r => r.confidence >= 0.58).length;
      if (highConf || suspicious >= Math.max(1, Math.floor(valid.length * 0.15))) {
        const avg = deepfakes.reduce((s, r) => s + r.confidence, 0) / deepfakes.length;
        return {
          type: "deepfake_detected",
          confidence: Math.max(avg, highConf?.confidence || 0),
          reason: deepfakes[0]?.reason || "",
        };
      }
    }

    // Confidence-weighted vote
    const scores = {};
    for (const r of valid) scores[r.type] = (scores[r.type] || 0) + (r.confidence || 0);
    const bestType = Object.keys(scores).reduce((a, b) => scores[a] >= scores[b] ? a : b);
    const matching = valid.filter(r => r.type === bestType);
    const avgConf = matching.reduce((s, r) => s + (r.confidence || 0), 0) / matching.length;

    return { type: bestType, confidence: avgConf, reason: matching[0]?.reason || "" };
  }

  /* ================================================================
     BADGE
  ================================================================ */

  function ensureBadge(player) {
    if (document.getElementById(BADGE_ID)) return;
    const pos = window.getComputedStyle(player).position;
    if (pos === "static") player.style.position = "relative";

    const badge = document.createElement("div");
    badge.id = BADGE_ID;
    badge.innerHTML = `
      <div class="badge-chip badge-loading">
        <span class="badge-icon">🔍</span>
        <span class="badge-label">Scanning…</span>
      </div>
      <div class="confidence-bar-wrapper">
        <div class="confidence-bar-fill" style="width:0%"></div>
      </div>
    `;
    player.appendChild(badge);
  }

  function removeBadge() {
    document.getElementById(BADGE_ID)?.remove();
  }

  /** Update badge while scanning: shows interim result + frame counter */
  function setBadgeScanning(framesReceived, framesTotal, interimResult) {
    const badge = document.getElementById(BADGE_ID);
    if (!badge) return;

    const chip = badge.querySelector(".badge-chip");
    const fill = badge.querySelector(".confidence-bar-fill");

    if (interimResult && framesReceived >= 1) {
      const info = LABELS[interimResult.type] || LABELS.error;
      const confPct = Math.round((interimResult.confidence || 0) * 100);
      chip.className = `badge-chip ${info.cls}`;
      chip.innerHTML = `
        <span class="badge-icon">${info.icon}</span>
        <span class="badge-label">${info.text} ${confPct}%
          <span class="badge-scanning-tag">&nbsp;·&nbsp;${framesReceived}/${framesTotal}</span>
        </span>
      `;
      fill.style.width = `${confPct}%`;
    } else {
      chip.className = "badge-chip badge-loading";
      chip.innerHTML = `
        <span class="badge-icon">🔍</span>
        <span class="badge-label">Scanning… (${framesReceived}/${framesTotal})</span>
      `;
      fill.style.width = `${Math.round((framesReceived / framesTotal) * 100)}%`;
    }
  }

  /** Lock in the final result after all frames are processed */
  function setFinalBadge(result) {
    const badge = document.getElementById(BADGE_ID);
    if (!badge) return;

    const isError = result.type === "error";
    const isOffline = result.type === "offline";
    const info = LABELS[result.type] || LABELS.error;
    const pct = Math.round((result.confidence || 0) * 100);

    const chip = badge.querySelector(".badge-chip");
    chip.className = `badge-chip ${info.cls}`;

    let labelText;
    if (isError && result.detail) labelText = result.detail;
    else if (isOffline) labelText = "Backend Offline";
    else labelText = `${info.text} &nbsp;${pct}%`;

    chip.innerHTML = `
      <span class="badge-icon">${info.icon}</span>
      <span class="badge-label">${labelText}</span>
    `;

    const fill = badge.querySelector(".confidence-bar-fill");
    fill.style.width = (isError || isOffline) ? "0%" : `${pct}%`;
  }

  /* ================================================================
     FRAME CAPTURE
  ================================================================ */

  function captureFrame(videoEl) {
    try {
      const canvas = document.createElement("canvas");
      // 224 px — native size for CLIP/ViT, minimises payload
      const scale = Math.min(1, 224 / Math.max(videoEl.videoHeight, 1));
      canvas.width = Math.round(videoEl.videoWidth * scale);
      canvas.height = Math.round(videoEl.videoHeight * scale);
      canvas.getContext("2d").drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.5);
    } catch (e) {
      console.warn("[AI Detector] Frame capture error:", e);
      return null;
    }
  }

  /* ================================================================
     METADATA
  ================================================================ */

  function getMetadata() {
    const isShorts = location.pathname.startsWith("/shorts/");
    let title = "", description = "";
    if (isShorts) {
      title = document.querySelector("yt-shorts-video-title-view-model h2 span")?.innerText || "";
      description = document.querySelector("ytd-structured-description-content-renderer yt-formatted-string")?.innerText
        || document.querySelector("#description-text")?.innerText || "";
    } else {
      title = document.querySelector("ytd-watch-metadata #title h1 yt-formatted-string")?.innerText || "";
      description = document.querySelector("ytd-watch-metadata #description-inline-expander yt-formatted-string")?.innerText
        || document.querySelector("#description-text")?.innerText || "";
    }
    return { title: title.trim(), description: description.trim().substring(0, 1000) };
  }

  /* ================================================================
     API CALL
  ================================================================ */

  async function sendFrame(b64Image, metadata) {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64Image, metadata }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();   // { type, confidence, reason }
  }

  /* ================================================================
     ADAPTIVE PARALLEL PROGRESSIVE ANALYSIS CYCLE
  ================================================================ */

  /**
   * One complete analysis cycle.
   * Frame count and interval are calculated fresh from video.duration
   * at the start of every cycle.
   */
  async function runCycle() {
    if (isRunning) return;

    const video = findVideo();
    if (!video) { scheduleCycle(RECYCLE_WAIT); return; }

    // Attach/recreate badge if player changed
    if (video !== lastVideoEl) {
      lastVideoEl = video;
      lastPlayerEl = findPlayerContainer(video);
    }
    if (!lastPlayerEl) { scheduleCycle(RECYCLE_WAIT); return; }
    if (video.paused || video.ended) { scheduleCycle(5000); return; }

    // ── Compute adaptive sampling plan from actual video duration ──
    const durationSec = isFinite(video.duration) && video.duration > 0
      ? video.duration
      : 90;    // fallback: treat unknown duration as ~90 s Short

    const { frameCount, intervalMs, windowMs } = computeSamplingPlan(durationSec);

    isRunning = true;
    accumulatedResults = [];
    framesCaptured = 0;
    ensureBadge(lastPlayerEl);

    const metadata = getMetadata();
    console.log(
      `[AI Detector] Cycle: duration=${durationSec.toFixed(0)}s → ` +
      `${frameCount} frames every ${(intervalMs / 1000).toFixed(1)}s ` +
      `(window=${(windowMs / 1000).toFixed(0)}s)`
    );

    // ── Inner callback: fires when each parallel request resolves ──
    function onResult(result, idx) {
      if (result.type !== "error") accumulatedResults.push(result);
      framesCaptured = Math.max(framesCaptured, idx + 1);
      const interim = accumulatedResults.length > 0 ? pickBestResult(accumulatedResults) : null;
      setBadgeScanning(framesCaptured, frameCount, interim);
      console.log(
        `[AI Detector] Frame ${idx + 1}/${frameCount}: ` +
        `${result.type} (${((result.confidence || 0) * 100).toFixed(0)}%) ` +
        (result.reason || "")
      );
    }

    // ── Capture frames & fire to backend in parallel ───────────────
    const framePromises = [];
    for (let i = 0; i < frameCount; i++) {
      const frameData = captureFrame(video);

      if (frameData) {
        const idx = i;
        framePromises.push(
          sendFrame(frameData, metadata)
            .then(res => { onResult(res, idx); return res; })
            .catch(err => {
              console.error(`[AI Detector] Frame ${idx + 1} error:`, err.message);
              const r = {
                type: "error", confidence: 0,
                detail: err.message.includes("Failed to fetch") ? "Backend Offline" : err.message,
              };
              onResult(r, idx);
              return r;
            })
        );
      } else {
        framesCaptured = Math.max(framesCaptured, i + 1);
      }

      // Wait before next capture to spread samples across playback time
      if (i < frameCount - 1) {
        await sleep(intervalMs);
        if (!isRunning) return;             // navigation cancelled cycle
        if (video.paused || video.ended) {
          console.log("[AI Detector] Video paused — awaiting remaining requests");
          break;
        }
      }
    }

    // ── Wait for ALL responses then lock in final badge ───────────
    await Promise.allSettled(framePromises);

    const finalResult = pickBestResult(accumulatedResults);
    if (finalResult) {
      setFinalBadge(finalResult);
      console.log(
        `[AI Detector] Final: ${finalResult.type} ` +
        `(${((finalResult.confidence || 0) * 100).toFixed(0)}%) ` +
        `from ${accumulatedResults.length}/${frameCount} valid frames`
      );
    }

    isRunning = false;
    scheduleCycle(RECYCLE_WAIT);
  }

  function scheduleCycle(delayMs) {
    if (cycleTimer) clearTimeout(cycleTimer);
    cycleTimer = setTimeout(runCycle, delayMs);
  }

  function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

  /* ================================================================
     OBSERVER – react to YouTube's SPA navigation
  ================================================================ */

  function handleNavigation() {
    isRunning = false;
    if (cycleTimer) clearTimeout(cycleTimer);
    cycleTimer = null;
    accumulatedResults = [];
    framesCaptured = 0;
    removeBadge();
    lastVideoEl = null;
    lastPlayerEl = null;
    setTimeout(runCycle, 1500);
  }

  let lastHref = location.href;
  const navObserver = new MutationObserver(() => {
    if (location.href !== lastHref) { lastHref = location.href; handleNavigation(); }
  });
  navObserver.observe(document.body, { childList: true, subtree: true });

  document.addEventListener("yt-navigate-finish", handleNavigation);

  // Initial start
  setTimeout(runCycle, 1500);

  console.log("[YouTube AI Media Detector] v1.3.0 — adaptive parallel sampling ✓");
})();
