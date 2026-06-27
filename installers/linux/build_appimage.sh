#!/usr/bin/env bash
# build_appimage.sh — crea un AppImage single-file para Linux x86_64.
#
# Prerequisitos:
#   - dist/mltutor/ producido por PyInstaller
#   - installers/mltutor.png (generado por create_icon.py, opcional pero recomendado)
#   - FUSE: sudo apt-get install libfuse2  (necesario para ejecutar AppImages)
#
# Uso desde la raíz del proyecto:
#   bash installers/linux/build_appimage.sh
#
# El AppImage se deja en la raíz del proyecto como mltutor-linux-x86_64.AppImage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist/mltutor"
APPDIR="$PROJECT_ROOT/dist/MLTutor.AppDir"
OUTPUT="$PROJECT_ROOT/mltutor-linux-x86_64.AppImage"
ICON_PNG="$PROJECT_ROOT/installers/mltutor.png"
APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
APPIMAGETOOL_BIN="$PROJECT_ROOT/dist/appimagetool"

if [[ ! -d "$DIST_DIR" ]]; then
    echo "✗ No se encontró dist/mltutor/. Ejecuta PyInstaller primero." >&2
    exit 1
fi

echo "▶ Preparando AppDir..."
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"

# Distribución PyInstaller dentro de la jerarquía FHS del AppImage
cp -r "$DIST_DIR" "$APPDIR/usr/bin/mltutor"

# AppRun
cp "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"

# .desktop
cp "$SCRIPT_DIR/mltutor.desktop" "$APPDIR/mltutor.desktop"

# Icono (requerido por appimagetool)
if [[ -f "$ICON_PNG" ]]; then
    cp "$ICON_PNG" "$APPDIR/mltutor.png"
else
    echo "⚠  No se encontró installers/mltutor.png; usando icono placeholder."
    python3 - "$APPDIR/mltutor.png" << 'PYEOF'
import struct, zlib, sys
from pathlib import Path

def pack_chunk(t, d):
    import struct, zlib
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xFFFFFFFF)

sig  = b'\x89PNG\r\n\x1a\n'
ihdr = pack_chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
row  = b'\x00' + b'\x29\x80\xb9' * 64   # RGB azul #2980b9
idat = pack_chunk(b'IDAT', zlib.compress(row * 64))
iend = pack_chunk(b'IEND', b'')
Path(sys.argv[1]).write_bytes(sig + ihdr + idat + iend)
PYEOF
fi

# ── appimagetool ──────────────────────────────────────────────────────────────
if command -v appimagetool &>/dev/null; then
    APPIMAGETOOL="appimagetool"
else
    if [[ ! -x "$APPIMAGETOOL_BIN" ]]; then
        echo "▶ Descargando appimagetool..."
        curl -L --fail -o "$APPIMAGETOOL_BIN" "$APPIMAGETOOL_URL"
        chmod +x "$APPIMAGETOOL_BIN"
    fi
    APPIMAGETOOL="$APPIMAGETOOL_BIN"
fi

echo "▶ Empaquetando AppImage..."
rm -f "$OUTPUT"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUTPUT"

echo "✓ AppImage listo: $OUTPUT"
echo "  Tamaño: $(du -sh "$OUTPUT" | cut -f1)"
