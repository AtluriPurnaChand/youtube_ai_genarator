# 🎥 YouTube AI Media Detector

[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](https://github.com/AtluriPurnaChand/youtube_ai_genarator)
[![Python](https://img.shields.io/badge/python-3.9+-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-orange.svg)](#)

A real-time, privacy-first AI system designed to detect and classify YouTube content directly on your local hardware. By combining a **Chrome Extension** and a high-performance **FastAPI Backend**, this tool identifies AI-generated videos, deepfakes, and CGI without sending your data to any cloud service.

---

## 🌟 Key Features

*   **100% Local Inference**: Your CPU/GPU handles all AI processing — no API keys required.
*   **Adaptive Sampling**: Scalable frame analysis based on video duration for precision and speed.
*   **Multi-Model Pipeline**: Uses CLIP (scene), MTCNN (face detection), and specialized ViT (deepfake detection).
*   **Progressive UI**: Instant feedback with live-updating scanning progress and results.
*   **Self-Healing Core**: Built-in runtime patches for environment issues (e.g., OneDrive metadata corruption).
*   **Premium Glassmorphism UI**: A sleek badge that integrates seamlessly into the YouTube player.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9+**
- **Google Chrome** (or any Chromium-based browser like Brave or Edge)
- **Git**

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AtluriPurnaChand/youtube_ai_genarator.git
cd youtube_ai_genarator
```

### 2. Backend Setup (Local AI Server)
The backend requires a Python environment to run the AI models.

```bash
cd backend
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```
*The server will start at `http://127.0.0.1:8000`.*

### 3. Chrome Extension Setup
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked**.
4. Select the `extension` folder from this repository.

---

## 🎮 How to Use
1. Ensure the **FastAPI Backend** is running.
2. Open any video on [YouTube](https://www.youtube.com).
3. A "Scanning" badge will appear automatically at the top-right of the video player.
4. Wait for the analysis to complete (usually 2–10 seconds depending on video length).
5. Click the badge for detailed classification results.

---

## 📏 Adaptive Sampling Strategy (v1.4.0)

| Video Type | Duration | Frames | Interval | Coverage |
|---|---|---|---|---|
| **Short** | < 30s | 4 | 5s | ~20s |
| **Short / Reel** | 30s – 90s | 6 | 8s | ~45s |
| **Standard** | 1.5m – 5m | 12 | 12s | ~2.5m |
| **Long-form** | 5m – 15m | 16 | 18s | ~5m |
| **Deep Scan** | > 15m | 20 | 25s | ~10m |

---

## 🛠 Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers (CLIP, ViT), MTCNN (Face Detection), PIL.
- **Extension**: Manifest V3, Content Scripts, Vanilla JS, CSS Glassmorphism.

---

## ❓ Troubleshooting

**Q: The badge says "Backend Offline".**
> A: Ensure your Python server is running (`python app.py`) and that it is accessible at `http://localhost:8000`. Check for firewall blocks.

**Q: Analysis is very slow.**
> A: Local AI inference depends on your CPU/GPU. First-time runs may be slower as models are downloaded to your local cache.

**Q: Metadata check isn't working.**
> A: Ensure the video page has fully loaded. The extension scans titles and descriptions for AI disclosures automatically.

---

## 📂 Project Structure
```text
youtube-ai-detector/
├── backend/            # FastAPI Server & AI Models
├── extension/          # Chrome Extension Files
├── generate_icons.py   # Utility script for icons
└── test_verification.py# Verification suite
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by [AtluriPurnaChand](https://github.com/AtluriPurnaChand)*
