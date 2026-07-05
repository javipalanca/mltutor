#!/usr/bin/env bash
# Publica una nueva release de MLTutor.
#
# Uso: ./release.sh X.Y.Z        (p. ej. ./release.sh 0.3.0)
#
# Hace todo el ciclo:
#   1. Actualiza la versión en pyproject.toml (única fuente de versión)
#   2. Commit + tag vX.Y.Z + push (dispara el CI)
#   3. Espera a que el CI compile los ejecutables de las 3 plataformas
#   4. Verifica que la release se ha publicado con todos los artefactos
#
# Requisitos: git, uv y gh (GitHub CLI autenticado).
set -euo pipefail

VERSION="${1:-}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Uso: ./release.sh X.Y.Z   (p. ej. ./release.sh 0.3.0)" >&2
  exit 1
fi
TAG="v$VERSION"

command -v gh >/dev/null || { echo "❌ Necesitas gh (GitHub CLI): brew install gh" >&2; exit 1; }
command -v uv >/dev/null || { echo "❌ Necesitas uv: https://docs.astral.sh/uv/" >&2; exit 1; }
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# --- Comprobaciones previas ---------------------------------------------
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "main" ]]; then
  echo "❌ Debes estar en main (estás en '$branch')" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Hay cambios sin commitear; haz commit o stash antes de publicar" >&2
  exit 1
fi
git pull --ff-only
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "❌ El tag $TAG ya existe" >&2
  exit 1
fi

# --- Versionar ------------------------------------------------------------
python3 - "$VERSION" <<'EOF'
import re, sys
version = sys.argv[1]
path = "pyproject.toml"
content = open(path).read()
updated = re.sub(r'(?m)^version = "[^"]+"$', f'version = "{version}"', content, count=1)
if content == updated and f'version = "{version}"' not in content:
    sys.exit("No se pudo actualizar la versión en pyproject.toml")
open(path, "w").write(updated)
EOF
uv lock

git add pyproject.toml uv.lock
if ! git diff --cached --quiet; then
  git commit -m "Release $TAG"
fi
git tag -a "$TAG" -m "MLTutor $TAG"
git push origin main "$TAG"
echo "✓ Tag $TAG pusheado; el CI está compilando los ejecutables"

# --- Esperar al workflow ---------------------------------------------------
echo "Esperando a que arranque el workflow..."
run_id=""
for _ in $(seq 1 30); do
  run_id=$(gh run list --workflow build-executables.yml --branch "$TAG" \
    --json databaseId -q '.[0].databaseId' 2>/dev/null || true)
  [[ -n "$run_id" ]] && break
  sleep 5
done
if [[ -z "$run_id" ]]; then
  echo "❌ No se encontró el workflow run para $TAG" >&2
  exit 1
fi
echo "Workflow: https://github.com/$REPO/actions/runs/$run_id"
gh run watch "$run_id" --exit-status

# --- Verificar la release --------------------------------------------------
echo
n_assets=$(gh release view "$TAG" --json assets -q '.assets | length')
if [[ "$n_assets" -lt 3 ]]; then
  echo "❌ La release solo tiene $n_assets artefactos (se esperaban 3)" >&2
  exit 1
fi
echo "✓ Release $TAG publicada con $n_assets artefactos:"
gh release view "$TAG" --json assets \
  -q '.assets[] | "  - \(.name) (\(.size/1048576|floor) MB)"'
echo
echo "🎉 https://github.com/$REPO/releases/tag/$TAG"
