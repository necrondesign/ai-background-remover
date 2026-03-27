# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Background Remover.

Cross-platform: works on macOS and Windows.

Usage:
    pyinstaller rmbg_remover.spec
"""

import os
import sys
import importlib

block_cipher = None

# Resolve CustomTkinter package path for bundling
ctk_path = os.path.dirname(importlib.import_module("customtkinter").__file__)

# Resolve HuggingFace model cache path
hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet")
if sys.platform == "win32":
    hf_cache_win = os.path.join(
        os.environ.get("USERPROFILE", ""), ".cache", "huggingface", "hub",
        "models--ZhengPeng7--BiRefNet",
    )
    if os.path.isdir(hf_cache_win):
        hf_cache = hf_cache_win

project_dir = os.path.dirname(os.path.abspath("rmbg_app.py"))

datas = [
    (ctk_path, "customtkinter"),
    # Текстуры слайдера
    (os.path.join(project_dir, "textures"), "textures"),
]

# Bundled tkdnd (Drag & Drop библиотека)
libs_tkdnd = os.path.join(project_dir, "libs", "tkdnd")
if os.path.isdir(libs_tkdnd):
    datas.append((libs_tkdnd, os.path.join("libs", "tkdnd")))

# Bundle tkinterdnd2 if available
try:
    tkdnd2_path = os.path.dirname(importlib.import_module("tkinterdnd2").__file__)
    datas.append((tkdnd2_path, "tkinterdnd2"))
except ImportError:
    pass

# Модель НЕ встраивается в билд — скачивается при первом запуске
# if os.path.isdir(hf_cache):
#     datas.append((hf_cache, "models--ZhengPeng7--BiRefNet"))

a = Analysis(
    ["rmbg_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PIL._tkinter_finder",
        "torch",
        "torchvision",
        "transformers",
        "kornia",
        "customtkinter",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "notebook",
        "jupyter",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Background Remover",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join("textures", "icon.icns" if sys.platform == "darwin" else "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AI Background Remover",
)

# macOS .app bundle (only on macOS)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AI Background Remover.app",
        icon=os.path.join("textures", "icon.icns"),
        bundle_identifier="com.andrushkevich.ai-bg-remover",
        info_plist={
            "CFBundleDisplayName": "AI Background Remover",
            "CFBundleShortVersionString": "1.2",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
