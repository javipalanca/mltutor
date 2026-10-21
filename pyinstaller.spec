# Portable Qt build: Windows/Linux single executable; macOS application bundle.
# uv run pyinstaller pyinstaller.spec --noconfirm
import os
import sys
import tomllib
from importlib.util import find_spec
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

for required in ('tensorflow', 'PySide6', 'sklearn', 'onnx', 'skl2onnx'):
    if find_spec(required) is None:
        raise RuntimeError(f'Falta {required} en el entorno de compilación. Comprueba las dependencias y la arquitectura de Python.')

project_root = SPECPATH
with open(os.path.join(project_root, 'pyproject.toml'), 'rb') as file:
    version = tomllib.load(file)['project']['version']

datas = [(os.path.join(project_root, 'mltutor', 'assets'), 'mltutor/assets'),
         (os.path.join(project_root, 'mltutor', 'dataset', 'data'), 'mltutor/dataset/data')]
for package in ('plotly', 'matplotlib', 'seaborn', 'certifi'):
    datas += collect_data_files(package)
if os.path.isdir(os.path.join(project_root, 'data')):
    datas.append((os.path.join(project_root, 'data'), 'data'))

# Markdown extensions and TensorFlow's dynamic imports are not all discoverable.
hiddenimports = collect_submodules('markdown.extensions')
hiddenimports += ['sklearn.utils._typedefs', 'sklearn.neighbors._quad_tree',
                  'sklearn.tree._utils', 'h5py', 'tensorflow', 'keras',
                  'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
                  'PySide6.QtPrintSupport']

a = Analysis([os.path.join(project_root, 'launcher_qt.py')],
    pathex=[project_root], binaries=[], datas=datas,
    hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['streamlit', 'webview', 'qtpy', 'tkinter', 'pytest', 'IPython',
              'PyQt5', 'PyQt6', 'PySide2'], noarchive=False)
pyz = PYZ(a.pure)

if sys.platform == 'darwin':
    # Apple app bundles already appear as a single application in Finder. Onedir
    # avoids unpacking TensorFlow at every launch and supports code signing.
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MLTutor',
              console=False, upx=False, argv_emulation=False)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MLTutor')
    app = BUNDLE(coll, name='MLTutor.app',
        icon=os.path.join(project_root, 'assets', 'icon.icns'),
        bundle_identifier='es.upv.mltutor',
        info_plist={'CFBundleName': 'MLTutor', 'CFBundleDisplayName': 'MLTutor',
                    'CFBundleShortVersionString': version, 'NSHighResolutionCapable': True})
else:
    # Onefile extracts only into the user's temporary directory; no installer,
    # administrative privileges, Python installation or Qt installation required.
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='MLTutor',
              console=False, upx=False,
              icon=os.path.join(project_root, 'assets', 'icon.ico') if sys.platform == 'win32' else None)
