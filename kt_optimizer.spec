# PyInstaller spec for stress-kt-estimator Windows exe
# Run: uv run pyinstaller kt_optimizer.spec

a = Analysis(
    ["./kt_optimizer/main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "kt_optimizer",
        "kt_optimizer.ui",
        "kt_optimizer.ui.main_window",
        "kt_optimizer.ui.result_panel",
        "kt_optimizer.ui.table_model",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "xhtml2pdf",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="kt-optimizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app, no console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
