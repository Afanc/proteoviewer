# -*- mode: python -*-

block_cipher = None

VERSION = "1.8.5" #until we properly package this

from PyInstaller.utils.hooks import copy_metadata
from glob import glob
from pathlib import Path
import sys

datas=copy_metadata('anndata') + copy_metadata('scanpy') + copy_metadata('scikit-learn') + copy_metadata('scikit-misc')

spec_dir = Path(sys.argv[0]).resolve().parent
resource_dir = spec_dir / "resources"

if not resource_dir.is_dir():
    raise SystemExit(f"Spec error: resources/ not found at {resource_dir}")

resource_files = list(resource_dir.glob("*"))
print("Resources picked up by spec:", [p.name for p in resource_files])

datas += [(str(p), "resources") for p in resource_files]

print("Resources picked up:", [p.name for p in resource_files])

a = Analysis(
    ['app.py'],
    pathex=[],
    datas=datas,
    hiddenimports=[
        'panel.io.server',
        'hatchling',
        'sklearn._cyutility',
        'fastcluster',
        'binascii',
        'skmisc',
    ],
    hookspath=[],
    runtime_hooks=['rthook_logging.py'],
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
    #name=f'proteoviewer-{VERSION}',
    name="proteoviewer",
)

import sys
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='ProteoViewer.app',
        icon=None,
        bundle_identifier='com.proteoviewer',
    )
