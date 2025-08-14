# -*- mode: python -*-
block_cipher = None
VERSION = "1.1.0" #until we package that thing

from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_tk_files
tk_datas, tk_bins = collect_tk_files()

datas=copy_metadata('anndata') + copy_metadata('scanpy') + copy_metadata('scikit-learn') + copy_metadata('scikit-misc') + tk_datas
a = Analysis(
    ['app.py'],           # your entry-point
    pathex=[],
    binaries=[tkbins],          # let PyInstaller auto-collect
    # copy_metadata returns a list of (src, dest) pairs—exactly what PyInstaller wants
    datas=datas,
    hiddenimports=[
        'panel.io.server',
        'hatchling',
        'PIL._tkinter_finder',
        'sklearn._cyutility',
        'fastcluster',
        'binascii',
        'skmisc',
        'tkinter',
        'tkinter.filedialog',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f'proteoviewer-{VERSION}',
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name=f'proteoviewer-{VERSION}',
)

import sys
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='ProteoViewer.app',
        icon=None,
        bundle_identifier='com.proteoviewer',
    )
