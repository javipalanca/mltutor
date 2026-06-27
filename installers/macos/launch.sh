#!/usr/bin/env bash
# Contents/MacOS/MLTutor — punto de entrada del .app bundle.
#
# Abre una ventana de Terminal.app con el proceso mltutor corriendo.
# El launcher (launcher_rich.py) muestra el panel Rich y abre el navegador.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MLTUTOR="$DIR/../Resources/mltutor/mltutor"

if [[ ! -x "$MLTUTOR" ]]; then
    osascript -e "display alert \"MLTutor\" message \"No se encontró el ejecutable:\\n$MLTUTOR\" as critical"
    exit 1
fi

# Abre una ventana nueva en Terminal.app ejecutando mltutor
osascript << APPLESCRIPT
tell application "Terminal"
    set w to do script "$MLTUTOR"
    set w's custom title to "MLTutor"
    activate
end tell
APPLESCRIPT
