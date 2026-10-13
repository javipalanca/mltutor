#!/usr/bin/env bash
# Publica una nueva release de MLTutor (versión Rust/Tauri).
#
# Uso: ./release.sh X.Y.Z[-rc.N] [--dry-run]
#      p. ej. ./release.sh 0.4.0-rc.2    o    ./release.sh 0.4.0 --dry-run
#
# Hace todo el ciclo:
#   1. Comprobaciones: rama (las estables solo desde main), árbol limpio y que el tag no exista
#   2. Actualiza la versión en Cargo.toml ([workspace.package]), src-tauri/tauri.conf.json
#      y ui/package.json, junto con Cargo.lock y ui/package-lock.json
#   3. Pide confirmación (y/N); después hace commit, crea el tag vX.Y.Z y hace push (dispara rust.yml)
#   4. Espera a que rust.yml compile y publique las 3 descargas
#   5. Verifica que la release tiene las 3 descargas y SHA256SUMS.txt
#
# --dry-run hace las comprobaciones y actualiza la versión para enseñar el diff, y después
# deja los archivos como estaban: no hace commit, ni tag, ni push.
#
# Requisitos: git, cargo, npm, python3 y gh (GitHub CLI autenticado; no hace falta con --dry-run).
set -euo pipefail

usage() {
  echo "Uso: ./release.sh X.Y.Z[-rc.N] [--dry-run]   (p. ej. ./release.sh 0.4.0-rc.2)" >&2
  exit 1
}

VERSION=""
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage ;;
    *) [[ -z "$VERSION" ]] || usage; VERSION="$arg" ;;
  esac
done
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]]; then
  usage
fi
TAG="v$VERSION"
WORKFLOW="rust.yml"
ASSETS=(mltutor-windows-x86_64.exe mltutor-macos-arm64.zip mltutor-linux-x86_64.AppImage SHA256SUMS.txt)
VERSION_FILES=(Cargo.toml Cargo.lock src-tauri/tauri.conf.json ui/package.json ui/package-lock.json)

cd "$(git rev-parse --show-toplevel)"

for tool in git cargo npm python3; do
  command -v "$tool" >/dev/null || { echo "❌ Falta $tool" >&2; exit 1; }
done
if [[ $DRY_RUN -eq 0 ]]; then
  command -v gh >/dev/null || { echo "❌ Necesitas gh (GitHub CLI): brew install gh" >&2; exit 1; }
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
fi

# --- Comprobaciones previas ---------------------------------------------
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "main" && "$VERSION" != *-rc.* ]]; then
  echo "❌ Las versiones estables requieren main; en '$branch' usa X.Y.Z-rc.N" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Hay cambios sin commitear; haz commit o stash antes de publicar" >&2
  exit 1
fi
if [[ "$branch" == "main" && $DRY_RUN -eq 0 ]]; then
  git pull --ff-only
fi
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "❌ El tag $TAG ya existe en local" >&2
  exit 1
fi
remote_status=0
git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1 || remote_status=$?
if [[ $remote_status -eq 0 ]]; then
  echo "❌ El tag $TAG ya existe en origin" >&2
  exit 1
elif [[ $remote_status -ne 2 ]]; then
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "❌ No se pudo consultar origin (código $remote_status)" >&2
    exit 1
  fi
  echo "⚠️  No se pudo consultar origin; se omite esa comprobación en --dry-run" >&2
fi

restore_version_files() {
  git checkout HEAD -- "${VERSION_FILES[@]}"
}
# Until the release commit exists, any exit (error, Ctrl-C at the prompt, "N") puts the
# version files back. The tree was clean at the start, so nothing else is lost.
PENDING_RESTORE=1
on_exit() {
  if [[ $PENDING_RESTORE -eq 1 ]]; then
    restore_version_files || true
  fi
}
trap on_exit EXIT

# --- Versionar ------------------------------------------------------------
python3 - "$VERSION" <<'EOF'
import json, re, sys

version = sys.argv[1]

def replace_once(path, pattern, replacement):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    updated, n = re.subn(pattern, replacement, content, count=1)
    if n != 1:
        sys.exit(f"No se pudo actualizar la versión en {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

# Cargo.toml: only the `version` key of the [workspace.package] table.
replace_once(
    "Cargo.toml",
    r'(?ms)(^\[workspace\.package\]\s*$(?:(?!^\[).)*?^version\s*=\s*)"[^"]*"',
    rf'\g<1>"{version}"',
)
# tauri.conf.json: the top-level key (two-space indentation), keeping the file layout.
replace_once("src-tauri/tauri.conf.json", r'(?m)^(  "version":\s*)"[^"]*"', rf'\g<1>"{version}"')
with open("src-tauri/tauri.conf.json", encoding="utf-8") as f:
    if json.load(f)["version"] != version:
        sys.exit("tauri.conf.json no ha quedado con la versión nueva")
EOF
# Updates ui/package.json and ui/package-lock.json (both root entries).
npm --prefix ui version "$VERSION" --no-git-tag-version --allow-same-version >/dev/null
# Rewrites only the workspace members in Cargo.lock.
cargo update --workspace --quiet

# Every file must now carry the same version (the `versions_are_in_sync` test checks it too).
if ! python3 - "$VERSION" <<'EOF'
import json, re, sys

version = sys.argv[1]
found = {}
cargo = open("Cargo.toml", encoding="utf-8").read()
m = re.search(r'(?ms)^\[workspace\.package\]\s*$(?:(?!^\[).)*?^version\s*=\s*"([^"]*)"', cargo)
found["Cargo.toml"] = m.group(1) if m else None
lock = open("Cargo.lock", encoding="utf-8").read()
m = re.search(r'(?m)^name = "mltutor"\nversion = "([^"]*)"', lock)
found["Cargo.lock"] = m.group(1) if m else None
found["src-tauri/tauri.conf.json"] = json.load(open("src-tauri/tauri.conf.json"))["version"]
found["ui/package.json"] = json.load(open("ui/package.json"))["version"]
pkg_lock = json.load(open("ui/package-lock.json"))
found["ui/package-lock.json"] = pkg_lock["version"]
found["ui/package-lock.json (packages)"] = pkg_lock["packages"][""]["version"]
bad = {k: v for k, v in found.items() if v != version}
for k, v in found.items():
    print(f"  {'✓' if v == version else '✗'} {k}: {v}")
sys.exit(1 if bad else 0)
EOF
then
  echo "❌ Las versiones no coinciden; se deshacen los cambios" >&2
  exit 1
fi

echo
git --no-pager diff --stat -- "${VERSION_FILES[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  git --no-pager diff -- Cargo.toml src-tauri/tauri.conf.json ui/package.json
  echo
  echo "✓ --dry-run: la versión se actualizaría a $VERSION (tag $TAG desde '$branch')."
  echo "  Archivos restaurados; no se ha hecho commit, tag ni push."
  exit 0
fi

# --- Confirmación ----------------------------------------------------------
echo
echo "Se va a crear el commit \"Release $TAG\" y el tag $TAG en '$branch',"
echo "y se hará push de ambos a origin ($REPO). Esto publica la release."
if [[ ! -t 0 ]]; then
  echo "❌ Hace falta una terminal interactiva para confirmar; se deshacen los cambios" >&2
  exit 1
fi
read -r -p "¿Continuar? [y/N] " answer
if [[ ! "$answer" =~ ^[yY]$ ]]; then
  echo "Cancelado; los archivos de versión se han restaurado."
  exit 1
fi

git add "${VERSION_FILES[@]}"
if ! git diff --cached --quiet; then
  git commit -m "Release $TAG"
fi
PENDING_RESTORE=0
git tag -a "$TAG" -m "MLTutor $TAG"
git push --atomic origin "$branch" "$TAG"
sha=$(git rev-parse HEAD)
echo "✓ Tag $TAG pusheado; $WORKFLOW está compilando las descargas"

# --- Esperar al workflow ---------------------------------------------------
echo "Esperando a que arranque el workflow..."
run_id=""
for _ in $(seq 1 60); do
  run_id=$(gh run list --workflow "$WORKFLOW" --event push --branch "$TAG" \
    --json databaseId,headSha \
    -q "map(select(.headSha == \"$sha\")) | .[0].databaseId // empty" 2>/dev/null || true)
  [[ -n "$run_id" ]] && break
  sleep 5
done
if [[ -z "$run_id" ]]; then
  echo "❌ No se encontró la ejecución de $WORKFLOW para $TAG" >&2
  exit 1
fi
echo "Workflow: https://github.com/$REPO/actions/runs/$run_id"
gh run watch "$run_id" --exit-status

# --- Verificar la release --------------------------------------------------
echo
names=$(gh release view "$TAG" --json assets -q '.assets[].name')
missing=0
for asset in "${ASSETS[@]}"; do
  if ! grep -qx "$asset" <<<"$names"; then
    echo "❌ Falta $asset en la release" >&2
    missing=1
  fi
done
[[ $missing -eq 0 ]] || exit 1

sums_dir=$(mktemp -d)
trap 'rm -rf "$sums_dir"' EXIT
gh release download "$TAG" --pattern SHA256SUMS.txt --dir "$sums_dir"
for asset in "${ASSETS[@]:0:3}"; do
  if ! grep -qE "^[0-9a-f]{64}  $asset\$" "$sums_dir/SHA256SUMS.txt"; then
    echo "❌ SHA256SUMS.txt no incluye $asset" >&2
    exit 1
  fi
done

prerelease=$(gh release view "$TAG" --json isPrerelease -q .isPrerelease)
expected=false
[[ "$VERSION" == *-* ]] && expected=true
if [[ "$prerelease" != "$expected" ]]; then
  echo "❌ isPrerelease=$prerelease y se esperaba $expected" >&2
  exit 1
fi

echo "✓ Release $TAG publicada (prerelease: $prerelease) con:"
gh release view "$TAG" --json assets \
  -q '.assets[] | "  - \(.name) (\(.size/1048576|floor) MB)"'
echo
echo "🎉 https://github.com/$REPO/releases/tag/$TAG"
