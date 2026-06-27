# PyInstaller spec para MLTutor (launcher + Streamlit + app)
# Build: uv run pyinstaller pyinstaller.spec
#
# Nota: mltutor/ se incluye como datos (Streamlit ejecuta app.py como
# fichero), por lo que PyInstaller no puede rastrear sus imports. Todas
# las librerías que usa la app deben forzarse aquí con collect_all.

import os
import sys as _sys

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all

project_root = os.path.abspath('.')

# Icono del ejecutable (solo relevante en Windows; en macOS va en el .app bundle)
_exe_icon = None
if _sys.platform == 'win32':
    _ico = os.path.join(project_root, 'installers', 'mltutor.ico')
    if os.path.exists(_ico):
        _exe_icon = _ico

datas = []
binaries = []
hiddenimports = []

# Paquetes que usa la app (importados desde app.py, que es un "dato")
for pkg in [
    'streamlit',
    'altair',
    'pyarrow',
    'sklearn',
    'scipy',
    'matplotlib',
    'seaborn',
    'numpy',
    'pandas',
    'PIL',
    'pydot',
    'onnx',
    'skl2onnx',
    'onnxconverter_common',
    'mpld3',
    'plotly',
    'joblib',
    'tensorflow',
    'keras',
    'rich',
    'dotenv',
]:
    try:
        ca_datas, ca_binaries, ca_hidden = collect_all(pkg)
        datas += ca_datas
        binaries += ca_binaries
        hiddenimports += ca_hidden
    except Exception:
        print(f'[spec] aviso: no se pudo recolectar {pkg}')

# Código fuente de la app como datos (sin caches); se añade en COLLECT,
# ya que Tree no es compatible con el formato de datas de Analysis
app_tree = Tree(
    os.path.join(project_root, 'mltutor'),
    prefix='mltutor',
    excludes=['__pycache__', '*.pyc', '.DS_Store'],
)

a = Analysis(
    ['launcher_rich.py'],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'pytest', 'IPython'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mltutor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=_exe_icon,
    console=True,  # mostrar consola para ver el output de rich
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    app_tree,
    strip=False,
    upx=False,
    name='mltutor',
)
