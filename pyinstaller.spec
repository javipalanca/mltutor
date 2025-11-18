# Minimal PyInstaller spec for mltutor launcher + Streamlit
# Build: uv run pyinstaller pyinstaller.spec

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree
import os

block_cipher = None

project_root = os.path.abspath('.')

datas = []
binaries = []
hiddenimports = []

# Collect third-party package data likely needed at runtime
for pkg in [
    'streamlit',
    'sklearn',
    'matplotlib',
    'seaborn',
    'numpy',
    'pandas',
    'pydot',
    'onnx',
    'skl2onnx',
    'mpld3',
    'plotly',
    'protobuf',
    'tensorflow',
    'rich',
]:
    try:
        ca_datas, ca_binaries, ca_hidden = collect_all(pkg)
        datas += ca_datas
        binaries += ca_binaries
        hiddenimports += ca_hidden
    except Exception:
        pass

# Add mltutor package manually
datas += [(os.path.join(project_root, 'mltutor'), 'mltutor')]

a = Analysis(
    ['launcher_rich.py'],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mltutor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mltutor',
)
