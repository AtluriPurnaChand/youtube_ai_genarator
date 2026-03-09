# 🎥 YouTube AI Media Detector

A real-time, AI-powered system consisting of a **Chrome Extension** and a **FastAPI Backend** to detect and classify YouTube video content. It identifies whether a video is authentic, AI-generated, a deepfake, or computer-generated (games/cartoons).

---

## 🚀 Key Features

*   **Real-time Analysis**: Captures and analyzes frames as you watch.
*   **Multi-Model Pipeline**: Uses CLIP for scene recognition and specialized ViTs for deepfake detection.
*   **Metadata Integration**: Scrapes video Title and Description to identify AI disclosures.
*   **Safety-First Aggregation**: Flags videos even if only a few frames show high-confidence manipulation.
*   **YouTube Shorts Support**: Optimized badge placement and metadata extraction for Shorts.
*   **Premium UI**: Glassmorphism-style badge with live progress bars.

---

## 🛠️ Performance Optimizations
The system supports the **native CLIP library** for significantly faster scene classification. 
> [!TIP]
> This optimization reduces the latency of the initial "Cartoon vs. Real" check on hardware with modern compilers.

---

## 📂 Folder Structure

```text
youtube-ai-detector/
├── backend/
│   ├── app.py            ← FastAPI endpoint
│   ├── analyzer.py       ← Core AI Pipeline (Logic Refined ✓)
│   └── requirements.txt  ← Dependencies (Native CLIP added ✓)
├── extension/
│   ├── manifest.json     ← Extension config
│   ├── content.js        ← Frame capture & aggregation logic
│   ├── style.css         ← Glassmorphism & Shorts UI fixes
│   └── icons/            ← Extension branding
└── generate_icons.py     ← Utility to create needed icon sizes
```

---

## ⚙️ Installation & Setup (VS Code)

### 1. Backend Setup
1.  Open **VS Code** and navigate to the `backend` folder.
2.  Create and activate a fresh virtual environment:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
3.  Install the required libraries:
    ```powershell
    pip install -r requirements.txt
    ```
4.  Launch the server:
    ```powershell
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```

### 2. Extension Setup
1.  Open **Google Chrome** and go to `chrome://extensions`.
2.  Enable **Developer Mode** (top right).
3.  Click **Load Unpacked** and select the `/extension` folder.
4.  Open any YouTube video or **Short** and look for the analysis badge!

---

## 🧠 AI Pipeline Logic (Refined)
The system uses a **multi-modal** approach for >80% accuracy:
1.  **Metadata Analysis**: Scrapes Title/Description for keywords (Sora, Kling, "AI generated", etc.).
2.  **Scene Classification (CLIP)**: Uses refined prompts for better synthetic texture detection.
3.  **Face Detection (MTCNN)**: Extracts human faces if present.
4.  **Deepfake Analysis (ViT)**: Specialized model (`dima806/deepfake_vs_real_image_detection`) for facial manipulation.
5.  **Weighted Signal Aggregation**: Combines all signals into a final classification.

---

## 📊 Classification Labels

| Label | Description |
|---|---|
| ✅ **Real Video** | Authentic, unmodified live footage. |
| 🤖 **AI Generated** | High-level synthetic generation (Sora, Kling, etc.). |
| 🎨 **Cartoon / Anime** | 2D or 3D animated content. |
| 🎮 **Video Game** | CGI or computer-generated gameplay. |
| 👤 **⚠ Deepfake** | Specific human face manipulation detected. |

---

## 🔗 Project Link
GitHub Repository: [https://github.com/AtluriPurnaChand/youtube_ai_genarator](https://github.com/AtluriPurnaChand/youtube_ai_genarator)

---
*Created as part of the AI-Powered Deepfake Detection for YouTube project.*