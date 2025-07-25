ProteoViewer Packaging Guide
1. Prerequisites

    Linux: any modern distro with Python 3.12

    Windows: Windows 11 (or later), Python 3.12 installed & on PATH

    Fish users on Linux: use activate.fish instead of activate for venvs

2. Create a clean virtualenv & install deps

# from your project root
python3 -m venv viewer_env
# on bash/zsh:
source viewer_env/bin/activate
# on fish:
#    source viewer_env/bin/activate.fish

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

3. proteoviewer.spec (the one that works)

# -*- mode: python -*-
import os
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# bundle metadata so importlib.metadata.version() works
datas = (
    copy_metadata('anndata')
  + copy_metadata('scanpy')
  + copy_metadata('scikit-learn')
)

a = Analysis(
    ['app.py'],     # your entry point
    pathex=[],
    # leave 'binaries' alone so PyInstaller auto-detects python312.dll, _ssl.pyd, etc.
    datas=datas,
    hiddenimports=[
        'panel.io.server',
        'hatchling',
        'PIL._tkinter_finder',  # for Matplotlib backends
        'sklearn._cyutility',   # scikit-learn C helper
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='proteoviewer',
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=True,
    name='proteoviewer',
)

    Why this works
    • No binaries=[] override: lets PyInstaller bundle the correct python312.dll, _ssl.pyd (with bundled OpenSSL), and all your .pyd/.dll extensions.
    • copy_metadata(...) lines ensure both anndata, scanpy, and scikit-learn can read their version info at runtime.
    • hiddenimports catches small helpers (PIL & scikit-learn) that PyInstaller’s hooks sometimes miss.

4. Build on Linux

# in your activated venv
rm -rf build dist
pyinstaller proteoviewer.spec
#   dist/proteoviewer/proteoviewer  +  dist/proteoviewer/_internal/
./dist/proteoviewer/proteoviewer

Navigate in your browser to http://localhost:5006/proteoviewer.
5. Build on Windows (local spare machine)

    Clone or pull your repo, activate a venv (PowerShell):

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Run PyInstaller:

rm -r build dist
pyinstaller proteoviewer.spec

Run in-place:

    cd dist\proteoviewer
    .\proteoviewer.exe

    or double-click in that folder (so it finds its _internal\).
