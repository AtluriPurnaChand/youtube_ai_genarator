import sys
import base64
from io import BytesIO
from PIL import Image

sys.path.append("backend")

from backend.analyzer import analyze_batch
from backend.evaluate import run_mock_test

def test_mock_evaluate():
    print("Running evaluate mock test...")
    run_mock_test()

def test_analyze_batch():
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Try analyzing a batch of 2 identical images
    print("Testing backend analyzer.py batch mode...")
    try:
        res = analyze_batch([b64, b64], metadata={"title": "Test Video"})
        print("Analysis Result:", res)
    except Exception as e:
        print("Analysis Failed:", e)

if __name__ == "__main__":
    test_mock_evaluate()
    test_analyze_batch()
