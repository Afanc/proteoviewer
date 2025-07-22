# -*- mode: python -*-
block_cipher = None

from PyInstaller.utils.hooks import copy_metadata

datas=copy_metadata('anndata') + copy_metadata('scanpy') + copy_metadata('scikit-learn') + copy_metadata('fastcluster')
a = Analysis(
    ['app.py'],           # your entry-point
    pathex=[],
    #binaries=[],          # let PyInstaller auto-collect
    # copy_metadata returns a list of (src, dest) pairs—exactly what PyInstaller wants
    datas=datas,
    hiddenimports=[
        'panel.io.server',
        'hatchling',
        'PIL._tkinter_finder',
        'sklearn._cyutility',
        'fastcluster',
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
    name='proteoviewer',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='proteoviewer',
)
