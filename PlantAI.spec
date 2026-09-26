# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


hiddenimports = []

hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("reportlab")


a = Analysis(
    [
        r"installer_build\app.py"
    ],

    pathex=[
        r"installer_build"
    ],

    binaries=[],

    datas=[
        (
            r"installer_build\models",
            r"models"
        ),
        (
            r"installer_build\templates",
            r"templates"
        ),
        (
            r"installer_build\static",
            r"static"
        ),
        (
            r"installer_build\utils",
            r"utils"
        )
    ],

    hiddenimports=hiddenimports,

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "notebook",
        "jupyter",
        "pytest"
    ],

    noarchive=False,
)

pyz = PYZ(
    a.pure
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PlantAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PlantAI",
    contents_directory="."
)
