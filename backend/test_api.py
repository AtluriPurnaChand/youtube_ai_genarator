import requests
import base64
from io import BytesIO
from PIL import Image


def test_health():
    """Ping the health endpoint first to verify backend is reachable."""
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if resp.status_code == 200:
            print("✅ Backend is online:", resp.json())
            return True
        else:
            print("❌ Backend returned non-200:", resp.status_code)
            return False
    except Exception as e:
        print("❌ Could not reach backend:", e)
        return False


def test_status():
    """Check which models are loaded."""
    try:
        resp = requests.get("http://127.0.0.1:8000/status", timeout=5)
        print("📊 Model status:", resp.json())
    except Exception as e:
        print("❌ Status check failed:", e)


def test_analyze():
    """Send a single base64-encoded frame to /analyze."""
    # Create a simple red test image
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    data = {
        "image": b64,   # ← single image string (field is 'image', not 'images')
        "metadata": {"title": "Test video", "description": "This is a test frame"},
    }
    print("\nSending POST request to /analyze …")
    try:
        response = requests.post("http://127.0.0.1:8000/analyze", json=data, timeout=60)
        print("Status code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error sending request:", e)


if __name__ == "__main__":
    if test_health():
        test_status()
        test_analyze()
