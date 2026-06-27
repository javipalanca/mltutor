#!/usr/bin/env python3
"""Genera el icono de MLTutor (PNG 512 px + ICO multi-size) usando Pillow.

Uso:
    python installers/create_icon.py [directorio_salida]

Por defecto guarda los ficheros en el mismo directorio que este script.
"""
import sys
from pathlib import Path


def create_icon(size: int = 512):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    m = size // 10
    r = size // 7

    # Fondo azul redondeado
    draw.rounded_rectangle([m, m, size - m, size - m], radius=r, fill=(41, 128, 185))

    # Texto "ML" blanco centrado
    font = None
    font_size = size * 2 // 5
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    text = "ML"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1]),
        text,
        fill="white",
        font=font,
    )
    return img


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = create_icon(512)

    png_path = out_dir / "mltutor.png"
    img.save(png_path, "PNG")
    print(f"PNG guardado: {png_path}")

    ico_path = out_dir / "mltutor.ico"
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icons = [img.resize(s, resample=3) for s in ico_sizes]  # 3 = LANCZOS
    icons[0].save(ico_path, format="ICO", append_images=icons[1:], sizes=ico_sizes)
    print(f"ICO guardado: {ico_path}")


if __name__ == "__main__":
    main()
