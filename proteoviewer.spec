# -*- mode: python -*-
block_cipher = None

from PyInstaller.utils.hooks import copy_metadata

datas=copy_metadata('anndata') + copy_metadata('scanpy') + copy_metadata('scikit-learn')
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
    console=True,
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

## -*- mode: python ; coding: utf-8 -*-
#import os
#from PyInstaller.utils.hooks import copy_metadata
#
#anndata_meta = copy_metadata('anndata')
#datas = anndata_meta + []
#
#a = Analysis(
#    ['app.py'],
#    pathex=[],
#    datas=datas,
#    hiddenimports=['panel.io.server',
#                   'hatchling'],
#    hookspath=[],
#    hooksconfig={},
#    runtime_hooks=[],
#    excludes=[],
#    noarchive=False,
#    optimize=0,
#)
#
## drop bundled OpenSSL & libcurl so we use the system’s versions instead
#import sys
#if sys.platform.startswith("linux"):
#    filtered_binaries = []
#    for name, path, typecode in a.binaries:
#        lib = os.path.basename(name)
#        if lib in ("libssl.so.3", "libcrypto.so.3", "libcurl.so.4"):
#            continue
#        filtered_binaries.append((name, path, typecode))
#    a.binaries = filtered_binaries
#
#pyz = PYZ(a.pure)
#
#exe = EXE(
#    pyz,
#    a.scripts,
#    a.binaries,
#    a.datas,
#    [],
#    name='proteoviewer',
#    debug=False,
#    bootloader_ignore_signals=False,
#    strip=False,
#    upx=True,
#    upx_exclude=[],
#    runtime_tmpdir=None,
#    console=True,
#    disable_windowed_traceback=False,
#    argv_emulation=False,
#    target_arch=None,
#    codesign_identity=None,
#    entitlements_file=None,
#)   
