# Sitio web de MLTutor (`site/`)

Página de presentación del proyecto en español: HTML, CSS y JavaScript (módulos ES) sin
dependencias ni paso de compilación. Está publicada (desde el 2026-10-21) como *project site* de
GitHub Pages de `javipalanca/mltutor`; con el dominio propio de la cuenta (`spadeagents.eu`) queda en
**https://spadeagents.eu/mltutor/**. Por eso **todas las rutas son relativas** (nada de `/…`) y
no hay `CNAME`: el dominio lo pone el sitio de usuario `javipalanca.github.io`.

## Estructura

- `index.html`: la página; `404.html`: autocontenida (estilos y script en línea) porque Pages la
  sirve para cualquier ruta inexistente, a cualquier profundidad; su enlace de vuelta se calcula
  como `../` relativo.
- `assets/css/styles.css`: temas claro y oscuro (`data-theme` en `<html>`, por defecto
  `prefers-color-scheme`, con botón que guarda la elección en `localStorage`), colores de
  `content/inicio/tarjetas.toml` y `prefers-reduced-motion`.
- `assets/js/`: `theme-init.js` (clásico, evita el parpadeo del tema), `main.js` y sus módulos:
  `knn-demo.js` (lienzo KNN del hero), `downloads.js` (SO y releases), `stats.js` (barras y
  contadores), `gallery.js` (capturas y visor), `tabs.js`, `theme.js`, `motion.js`.
- `assets/img/capturas/`: capturas de `docs/migration/capturas/rust/` en WebP (1400 px y 640 px).
  Iconos: `favicon.ico`, `favicon-32.png`, `apple-touch-icon.png` e `icono-96.webp`, de
  `src-tauri/icons/`; `og-image.jpg` (1200×630) sale de la captura de inicio.

Las cifras (tamaños, tiempos) se copian de `README.md` y `docs/migration/benchmarks.md`: si
cambian allí, hay que cambiarlas en `index.html` (atributos `data-python`/`data-rust` y textos).

## Descargas

`downloads.js` consulta en el navegador `https://api.github.com/repos/javipalanca/mltutor/releases`
(caché de 10 min en `sessionStorage`) y elige la release más reciente, no borrador, con tag
**≥ v0.4.0** que tenga `mltutor-windows-x86_64.exe`, `mltutor-macos-arm64.zip` o
`mltutor-linux-x86_64.AppImage`. El filtro por versión es necesario: las releases Python
`v0.3.0-rc.N` (y la `v0.1.0`) usan algunos de esos mismos nombres. Si no la hay, muestra
«Próximamente» y enlaza a la página de Releases; los errores y el límite de consultas de GitHub
también enlazan a esa página.

Para probar los estados sin llamar a la API: `?releases=none`, `?releases=rust`,
`?releases=error`, `?releases=ratelimit` u `?releases=offline`.

## Vista previa local (con el prefijo `/mltutor/`)

```sh
tmp=$(mktemp -d) && ln -s "$PWD/site" "$tmp/mltutor"
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$tmp"
# abre http://127.0.0.1:8765/mltutor/
```

## Publicar (solo con permiso explícito)

Pages sirve la rama **`gh-pages`** (fuente *Deploy from a branch*, carpeta `/`), que contiene solo
el contenido de `site/`. Así no hace falta tocar `main` ni ningún workflow. Para actualizar la web
después de cambiar `site/` en la rama de trabajo:

```sh
git subtree split --prefix site -b gh-pages-nuevo
git push origin gh-pages-nuevo:gh-pages   # Pages vuelve a publicar en 1–5 minutos
git branch -D gh-pages-nuevo
```

Si el push se rechaza por historia distinta, se puede forzar (`--force`): la rama solo contiene la
web generada. Comprueba después https://spadeagents.eu/mltutor/ y
https://spadeagents.eu/mltutor/no-existe (404).

`og:image` es relativa; algunas redes sociales solo aceptan URL absolutas. Si las vistas previas al
compartir salen sin imagen, cambia `og:image` y `twitter:image` a
`https://spadeagents.eu/mltutor/assets/img/og-image.jpg`.
