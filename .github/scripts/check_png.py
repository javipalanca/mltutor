"""Checks that a `mltutor --screenshot` PNG is a real capture, not a blank window.

Standard library only (runs on the three CI runners without installing anything).
Fails (exit 1) if the file is small, the image is smaller than --min-width/--min-height,
or it is (almost) a single colour.

    python check_png.py out.png [--min-width 800] [--min-height 500] [--summary FILE]
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

SIGNATURE = b"\x89PNG\r\n\x1a\n"
# colour type -> samples per pixel (8-bit only; palette indices count as colours)
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def decode(data: bytes) -> tuple[int, int, int, list[bytes]]:
    """Width, height, bytes per pixel and the unfiltered scanlines of an 8-bit PNG."""
    if not data.startswith(SIGNATURE):
        raise ValueError("no es un PNG")
    pos, idat, header = len(SIGNATURE), [], None
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise ValueError("falta IHDR")
    width, height, depth, colour, _, _, interlace = header
    if depth != 8 or colour not in CHANNELS or interlace:
        raise ValueError(f"formato no soportado (profundidad {depth}, tipo {colour}, entrelazado {interlace})")
    bpp = CHANNELS[colour]
    raw = zlib.decompress(b"".join(idat))
    stride = width * bpp
    rows: list[bytes] = []
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1 : start + 1 + stride])
        if kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b, c = prev[i], prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif kind != 0:
            raise ValueError(f"filtro {kind} desconocido en la fila {y}")
        rows.append(bytes(line))
        prev = line
    return width, height, bpp, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("png", type=Path)
    parser.add_argument("--min-width", type=int, default=800)
    parser.add_argument("--min-height", type=int, default=500)
    parser.add_argument("--min-bytes", type=int, default=20_000)
    parser.add_argument("--summary", type=Path, help="Markdown file to append a line to")
    args = parser.parse_args()

    def report(line: str, errors: list[str]) -> int:
        if args.summary:
            with args.summary.open("a", encoding="utf-8") as summary:
                summary.write(f"- {'❌' if errors else '✅'} {line}\n")
        for error in errors:
            print(f"::error::{args.png.name} {error}")
        return 1 if errors else 0

    try:
        data = args.png.read_bytes()
        width, height, bpp, rows = decode(data)
    except (OSError, ValueError, zlib.error, IndexError, struct.error) as error:
        # A truncated or corrupt file must fail with a readable annotation, not a traceback.
        return report(f"{args.png.name}: no se puede leer", [f"no es un PNG legible: {error}"])
    # Every 4th pixel of every 4th row is plenty to tell a page from a blank window.
    colours = Counter(row[x : x + bpp] for row in rows[::4] for x in range(0, len(row), 4 * bpp))
    sampled = sum(colours.values())
    dominant = colours.most_common(1)[0][1] / sampled
    line = (
        f"{args.png.name}: {width}×{height} px, {len(data):,} bytes, "
        f"{len(colours)} colores, color dominante {dominant:.1%}"
    )
    print(line)
    errors = []
    if len(data) < args.min_bytes:
        errors.append(f"ocupa menos de {args.min_bytes} bytes")
    if width < args.min_width or height < args.min_height:
        errors.append(f"es menor que {args.min_width}×{args.min_height}")
    if len(colours) < 32 or dominant > 0.97:
        errors.append("es casi de un solo color (¿ventana en blanco?)")
    return report(line, errors)


if __name__ == "__main__":
    sys.exit(main())
