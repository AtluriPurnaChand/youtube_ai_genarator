"""
generate_icons.py  –  Creates the three PNG icons required by the Chrome extension.
Run once before loading the extension in Chrome:

    python generate_icons.py
"""

import os
from PIL import Image, ImageDraw, ImageFont

sizes = [16, 48, 128]
out_dir = os.path.join(os.path.dirname(__file__), "extension", "icons")
os.makedirs(out_dir, exist_ok=True)

for size in sizes:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient-like background circle
    draw.ellipse([0, 0, size - 1, size - 1], fill=(139, 92, 246, 230))

    # Draw a tiny robot emoji / magnifier placeholder
    margin = size // 6
    inner  = size - 2 * margin
    draw.ellipse(
        [margin + inner // 4, margin + inner // 4,
         margin + 3 * inner // 4, margin + 3 * inner // 4],
        outline=(255, 255, 255, 255),
        width=max(1, size // 12),
    )
    # Handle of magnifier
    hw = max(1, size // 10)
    draw.line(
        [margin + 3 * inner // 4, margin + 3 * inner // 4,
         margin + inner - hw, margin + inner - hw],
        fill=(255, 255, 255, 255),
        width=hw,
    )

    path = os.path.join(out_dir, f"icon{size}.png")
    img.save(path)
    print(f"Saved {path}")

print("Icons generated successfully.")
