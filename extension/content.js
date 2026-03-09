/**
 * YouTube AI Media Detector – Content Script
 *
 * - Works on regular YouTube videos  (/watch?v=...)
 * - Works on YouTube Shorts          (/shorts/...)
 * - Captures 5 frames per second for 5 seconds (25 frames total)
 * - Sends each frame to the local FastAPI backend
 * - Shows a "Analyzing…" badge while collecting, then ONE final result
 * - Badge fixed to the TOP-RIGHT corner of the video player
 */

; (function () {
  "use strict";

  /* ================================================================
     CONFIG
  ================================================================ */
  const API_URL = "http://localhost:8000/analyze";
  const ANALYSIS_WINDOW = 5000;   // Total analysis window: 5 seconds
  const SAMPLE_COUNT = 25;       // 5 frames per second (25 total)
  const BADGE_ID = "yt-ai-detector-badge";

  /* ================================================================
     LABEL MAP  (backend type → display text + icon + CSS class)
  ================================================================ */
  const LABELS = {
    real_video: { text: "Real Video", icon: "✅", cls: "badge-real" },
    ai_generated: { text: "AI Generated", icon: "🤖", cls: "badge-ai" },
    cartoon_animation: { text: "Cartoon / Anime", icon: "🎨", cls: "badge-cartoon" },
    video_game: { text: "Video Game", icon: "🎮", cls: "badge-game" },
    deepfake_detected: { text: "⚠ Deepfake", icon: "👤", cls: "badge-deepfake" },
    error: { text: "Detection Error", icon: "❓", cls: "badge-error" },
  };

  /* ================================================================
     STATE
  ================================================================ */
  let analysisTimer = null;   // setTimeout handle for next analysis cycle
  let lastVideoEl = null;
  let lastPlayerEl = null;
  let isAnalyzing = false;  // Guard: don't start a new cycle mid-analysis

  /* ================================================================
     UTILITIES
  ================================================================ */

  /** Find the active <video> element on the page */
  function findVideo() {
    const candidates = Array.from(document.querySelectorAll("video"));
    return candidates
      .filter(v => v.readyState >= 2 && v.videoWidth > 0)
      .sort((a, b) => (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight))[0]
      || candidates[0]
      || null;
  }

  /** Find the nearest ancestor that is the YT player container */
  function findPlayerContainer(videoEl) {
    let el = videoEl?.parentElement;
    while (el) {
      // Regular YT player
      if (el.id === "movie_player" || el.classList.contains("html5-video-player")) {
        return el;
      }
      // Shorts player (desktop/mobile layout)
      if (
        el.tagName === "YTD-SHORTS" ||
        el.classList.contains("reel-video-in-sequence") ||
        el.id === "shorts-container" ||
        el.classList.contains("video-stream")
      ) {
        // Find the inner-most container that isn't the video element itself
        return el;
      }
      el = el.parentElement;
    }
    return videoEl?.parentElement || document.body;
  }

  /** Simple majority-vote over an array of result objects */
  function pickBestResult(results) {
    if (!results.length) return { type: "error", confidence: 0 };

    // ── SAFETY-FIRST AGGREGATION ──
    // If any frame is flagged as deepfake with >= 75% confidence,
    // OR if more than 15% of frames (e.g. 4/25) are flagged with >= 60% confidence,
    // we prioritize "deepfake_detected" even if the majority says real.
    const deepfakes = results.filter(r => r.type === "deepfake_detected");
    const highConfDeepfake = deepfakes.find(r => r.confidence >= 0.75);
    const suspiciousCount = deepfakes.filter(r => r.confidence >= 0.60).length;

    if (highConfDeepfake || (suspiciousCount >= Math.max(1, SAMPLE_COUNT * 0.15))) {
      const avgDeepfakeConf = deepfakes.reduce((s, r) => s + (r.confidence || 0), 0) / (deepfakes.length || 1);
      return { type: "deepfake_detected", confidence: Math.max(avgDeepfakeConf, highConfDeepfake?.confidence || 0) };
    }

    // Otherwise, do standard confidence-weighted voting
    const scores = {};
    for (const r of results) {
      if (!scores[r.type]) scores[r.type] = 0;
      scores[r.type] += (r.confidence || 0);
    }

    const bestType = Object.keys(scores).reduce((a, b) => scores[a] >= scores[b] ? a : b);
    const matching = results.filter(r => r.type === bestType);
    const avgConf = matching.reduce((s, r) => s + (r.confidence || 0), 0) / matching.length;

    return { type: bestType, confidence: avgConf };
  }

  /* ================================================================
     BADGE
  ================================================================ */

  function createBadge(player) {
    removeBadge();

    const pos = window.getComputedStyle(player).position;
    if (pos === "static") player.style.position = "relative";

    const badge = document.createElement("div");
    badge.id = BADGE_ID;
    badge.innerHTML = `
      <div class="badge-chip badge-loading">
        <span class="badge-icon">🔍</span>
        <span class="badge-label">Analyzing…</span>
      </div>
      <div class="confidence-bar-wrapper">
        <div class="confidence-bar-fill" style="width:0%"></div>
      </div>
    `;
    player.appendChild(badge);
    return badge;
  }

  function removeBadge() {
    document.getElementById(BADGE_ID)?.remove();
  }

  function setBadgeAnalyzing(currentSample) {
    const badge = document.getElementById(BADGE_ID);
    if (!badge) return;
    const chip = badge.querySelector(".badge-chip");
    chip.className = "badge-chip badge-loading";
    chip.innerHTML = `
      <span class="badge-icon">🔍</span>
      <span class="badge-label">Analyzing… (${currentSample}/${SAMPLE_COUNT})</span>
    `;
    const fill = badge.querySelector(".confidence-bar-fill");
    fill.style.width = `${Math.round((currentSample / SAMPLE_COUNT) * 100)}%`;
  }

  function updateBadge(result) {
    const badge = document.getElementById(BADGE_ID);
    if (!badge) return;

    const info = LABELS[result.type] || LABELS.error;
    const pct = Math.round((result.confidence || 0) * 100);

    const chip = badge.querySelector(".badge-chip");
    chip.className = `badge-chip ${info.cls}`;
    chip.innerHTML = `
      <span class="badge-icon">${info.icon}</span>
      <span class="badge-label">${info.text} &nbsp;${pct}%</span>
    `;

    const fill = badge.querySelector(".confidence-bar-fill");
    fill.style.width = `${pct}%`;
  }

  /* ================================================================
     FRAME CAPTURE
  ================================================================ */

  function captureFrame(videoEl) {
    try {
      const canvas = document.createElement("canvas");
      // Use 224px directly — native size for CLIP/ViT models, saves bandwidth and CPU
      const scale = Math.min(1, 224 / Math.max(videoEl.videoHeight, 1));
      canvas.width = Math.round(videoEl.videoWidth * scale);
      canvas.height = Math.round(videoEl.videoHeight * scale);

      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.5);
    } catch (e) {
      console.warn("[AI Detector] Frame capture error:", e);
      return null;
    }
  }

  /* ================================================================
     METADATA EXTRACTION
  ================================================================ */

  /** Extract title and description for both regular videos and shorts */
  function getMetadata() {
    const isShorts = location.pathname.startsWith("/shorts/");
    let title = "";
    let description = "";

    if (isShorts) {
      // Shorts metadata
      title = document.querySelector("yt-shorts-video-title-view-model h2 span")?.innerText || "";
      // Description is often in a specific renderer
      description = document.querySelector("ytd-structured-description-content-renderer yt-formatted-string")?.innerText || "";

      // If description is empty, try to find the 'more' button overlay content if it exists
      if (!description) {
        description = document.querySelector("#description-text")?.innerText || "";
      }
    } else {
      // Regular video metadata
      title = document.querySelector("ytd-watch-metadata #title h1 yt-formatted-string")?.innerText || "";
      // Get the description text (handles collapsed/expanded)
      description = document.querySelector("ytd-watch-metadata #description-inline-expander yt-formatted-string")?.innerText
        || document.querySelector("#description-text")?.innerText
        || "";
    }

    return {
      title: title.trim(),
      description: description.trim().substring(0, 1000), // Limit length for speed
    };
  }

  /* ================================================================
     API CALL
  ================================================================ */

  async function analyzeFrame(b64Image, metadata) {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64Image, metadata: metadata }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();   // { type, confidence }
  }

  /* ================================================================
     MAIN ANALYSIS CYCLE  (runs once every ~10 seconds)
  ================================================================ */

  async function runAnalysisCycle() {
    if (isAnalyzing) return;

    const video = findVideo();
    if (!video) {
      scheduleNextCycle();
      return;
    }

    // Re-attach badge if the video element changed
    if (video !== lastVideoEl) {
      lastVideoEl = video;
      lastPlayerEl = findPlayerContainer(video);
      if (lastPlayerEl) createBadge(lastPlayerEl);
    }

    if (!lastPlayerEl) {
      scheduleNextCycle();
      return;
    }

    // Skip capture while paused / ended
    if (video.paused || video.ended) {
      scheduleNextCycle();
      return;
    }

    isAnalyzing = true;

    // ── Collect SAMPLE_COUNT frames at strict intervals ──
    const interval = ANALYSIS_WINDOW / (SAMPLE_COUNT - 1 || 1);
    const pendingRequests = [];

    const metadata = getMetadata();
    console.log("[AI Detector] Analyzing with metadata:", metadata);

    for (let i = 0; i < SAMPLE_COUNT; i++) {
      setBadgeAnalyzing(i + 1);

      const frame = captureFrame(video);
      if (frame) {
        // Fire request without awaiting it here
        const request = analyzeFrame(frame, metadata)
          .catch(err => {
            console.error("[AI Detector] Backend error:", err.message);
            if (err.message.includes("Failed to fetch")) {
              return { type: "error", confidence: 0, detail: "Backend Offline" };
            }
            return { type: "error", confidence: 0 };
          });
        pendingRequests.push(request);
      }

      // Wait exactly 'interval' to ensure capture fits the 5s window
      if (i < SAMPLE_COUNT - 1) {
        await sleep(interval);
      }
    }

    // ── Wait for all backend responses to finish ──
    const results = await Promise.all(pendingRequests);

    // ── Show the single aggregated result ──
    const finalResult = pickBestResult(results);
    updateBadge(finalResult);

    isAnalyzing = false;

    // Schedule the next full cycle
    scheduleNextCycle();
  }

  function scheduleNextCycle() {
    if (analysisTimer) clearTimeout(analysisTimer);
    // Next analysis starts after the full window completes
    analysisTimer = setTimeout(runAnalysisCycle, ANALYSIS_WINDOW);
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /* ================================================================
     OBSERVER – react to YouTube's SPA navigation
  ================================================================ */

  function handleNavigation() {
    isAnalyzing = false;
    if (analysisTimer) clearTimeout(analysisTimer);
    analysisTimer = null;

    removeBadge();
    lastVideoEl = null;
    lastPlayerEl = null;

    // Wait briefly for the new player to mount, then begin
    setTimeout(runAnalysisCycle, 1500);
  }

  // YouTube is a SPA – listen for URL changes
  let lastHref = location.href;
  const navObserver = new MutationObserver(() => {
    if (location.href !== lastHref) {
      lastHref = location.href;
      handleNavigation();
    }
  });
  navObserver.observe(document.body, { childList: true, subtree: true });

  // Also react to yt-navigate-finish custom event
  document.addEventListener("yt-navigate-finish", handleNavigation);

  // Initial start (short delay so the player can load)
  setTimeout(runAnalysisCycle, 1500);

  console.log("[YouTube AI Media Detector] Content script loaded ✓");
})();
