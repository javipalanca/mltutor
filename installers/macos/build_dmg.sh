#!/usr/bin/env bash
# build_dmg.sh — crea MLTutor.app y el DMG distribuible para macOS.
#
# Prerequisitos:
#   - dist/mltutor/ producido por PyInstaller
#   - installers/mltutor.png  (generado por create_icon.py, opcional)
#
# Uso desde la raíz del proyecto:
#   bash installers/macos/build_dmg.sh
#
# El DMG se deja en la raíz del proyecto como mltutor-macos-arm64.dmg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist/mltutor"
APP_NAME="MLTutor"
APP_BUNDLE="$PROJECT_ROOT/dist/${APP_NAME}.app"
DMG_PATH="$PROJECT_ROOT/mltutor-macos-arm64.dmg"
ICON_PNG="$PROJECT_ROOT/installers/mltutor.png"

VERSION="$(python3 -c "
import tomllib, pathlib
data = tomllib.load(open('$PROJECT_ROOT/pyproject.toml', 'rb'))
print(data['project']['version'])
")"

echo "▶ Versión: $VERSION"
echo "▶ Construyendo ${APP_NAME}.app..."

# ── Limpieza ──────────────────────────────────────────────────────────────────
rm -rf "$APP_BUNDLE"
rm -f  "$DMG_PATH"

# ── Estructura .app ───────────────────────────────────────────────────────────
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copiar distribución PyInstaller
cp -r "$DIST_DIR" "$APP_BUNDLE/Contents/Resources/mltutor"

# Info.plist con versión sustituida
sed "s/%%VERSION%%/$VERSION/g" "$SCRIPT_DIR/Info.plist" \
    > "$APP_BUNDLE/Contents/Info.plist"

# Launcher script como ejecutable del bundle
cp "$SCRIPT_DIR/launch.sh" "$APP_BUNDLE/Contents/MacOS/MLTutor"
chmod +x "$APP_BUNDLE/Contents/MacOS/MLTutor"

# ── Icono .icns ───────────────────────────────────────────────────────────────
if [[ -f "$ICON_PNG" ]]; then
    echo "▶ Generando AppIcon.icns..."
    ICONSET="$(mktemp -d)/MLTutor.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512; do
        sips -z $size $size "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
        double=$((size * 2))
        sips -z $double $double "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
    rm -rf "$(dirname "$ICONSET")"
else
    echo "⚠  No se encontró mltutor.png; el .app usará el icono por defecto."
fi

# ── Crear DMG ─────────────────────────────────────────────────────────────────
echo "▶ Creando $DMG_PATH..."

TMP_DMG_DIR="$(mktemp -d)"
cp -r "$APP_BUNDLE" "$TMP_DMG_DIR/${APP_NAME}.app"
# Enlace a /Applications para arrastrar e instalar
ln -s /Applications "$TMP_DMG_DIR/Applications"

hdiutil create \
    -srcfolder  "$TMP_DMG_DIR" \
    -volname    "$APP_NAME" \
    -fs         HFS+ \
    -format     UDZO \
    -imagekey   zlib-level=9 \
    "$DMG_PATH"

rm -rf "$TMP_DMG_DIR"

echo "✓ DMG listo: $DMG_PATH"
