# PyInstaller spec para MLTutor (launcher + Streamlit + app)
# Build: uv run pyinstaller pyinstaller.spec
#
# Nota: mltutor/ se incluye como datos (Streamlit ejecuta app.py como
# fichero), por lo que PyInstaller no puede rastrear sus imports. Todas
# las librerías que usa la app deben forzarse aquí con collect_all.

import os
import sys

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all

project_root = os.path.abspath('.')

datas = []
binaries = []
hiddenimports = []

# Paquetes que usa la app (importados desde app.py, que es un "dato")
collect_pkgs = [
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
    # pywebview (ventana nativa de escritorio)
    'webview',
]

for pkg in collect_pkgs:
    try:
        ca_datas, ca_binaries, ca_hidden = collect_all(pkg)
        datas += ca_datas
        binaries += ca_binaries
        hiddenimports += ca_hidden
    except Exception:
        print(f'[spec] aviso: no se pudo recolectar {pkg}')

# Backend de pywebview en Linux: Qt WebEngine vía qtpy/PySide6 (importados
# dinámicamente, PyInstaller no los detecta solo)
if sys.platform.startswith('linux'):
    hiddenimports += [
        'qtpy',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtNetwork',
        'PySide6.QtWebChannel',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtPrintSupport',
    ]

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
    # App de escritorio: sin consola en Windows (la salida va a
    # ~/.mltutor/mltutor.log). En Linux se mantiene la consola porque el
    # binario se lanza desde terminal y sirve de diagnóstico; en macOS el
    # flag solo afecta al binario suelto (la distribución es MLTutor.app).
    console=sys.platform != 'win32',
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

# En macOS, además del directorio dist/mltutor se genera un bundle
# MLTutor.app con doble clic nativo (sin terminal)
if sys.platform == 'darwin':
    from PyInstaller.building.osx import BUNDLE

    app = BUNDLE(
        coll,
        name='MLTutor.app',
        icon=None,
        bundle_identifier='es.upv.mltutor',
        info_plist={
            'CFBundleName': 'MLTutor',
            'CFBundleDisplayName': 'MLTutor',
            'CFBundleShortVersionString': '0.1.0',
            'NSHighResolutionCapable': True,
        },
    )
