#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build.sh — Build AI Background Remover .app (venv-based bundle)
#
# Usage:
#   ./scripts/build.sh              # Build .app only
#   ./scripts/build.sh --dmg        # Build .app + create DMG
# ---------------------------------------------------------------------------

set -euo pipefail

APP_NAME="AI Background Remover"
APP_VERSION="1.2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_DIR/dist"
# Используем venv если есть, иначе системный python (для CI)
if [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
    VENV_PY="$PROJECT_DIR/.venv/bin/python3"
else
    VENV_PY="$(which python3)"
fi

CREATE_DMG=false
for arg in "$@"; do
    case "$arg" in
        --dmg) CREATE_DMG=true ;;
    esac
done

echo "=== Building $APP_NAME v$APP_VERSION ==="

# -------------------------------------------------------------------------
# 1. Assemble .app bundle
# -------------------------------------------------------------------------
echo "→ Assembling .app bundle..."

APP_DIR="$DIST_DIR/$APP_NAME.app"
rm -rf "$APP_DIR"

CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
BUNDLE_VENV="$RESOURCES/venv"

mkdir -p "$MACOS" "$RESOURCES"

# Copy app source
cp "$PROJECT_DIR/rmbg_app.py" "$RESOURCES/"

# Copy textures (PNG-текстуры слайдера)
cp -a "$PROJECT_DIR/textures" "$RESOURCES/textures"

# Copy bundled tkdnd (Drag & Drop библиотека)
if [ -d "$PROJECT_DIR/libs" ]; then
    cp -a "$PROJECT_DIR/libs" "$RESOURCES/libs"
fi

# Create venv inside bundle and install deps
echo "→ Creating bundle venv + installing dependencies..."
"$VENV_PY" -m venv "$BUNDLE_VENV"
"$BUNDLE_VENV/bin/python3" -m pip install -q -r "$PROJECT_DIR/requirements.txt"

# -------------------------------------------------------------------------
# 2. Download model if needed (uses bundle venv where transformers is installed)
# -------------------------------------------------------------------------
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet"
if [ ! -d "$MODEL_CACHE" ]; then
    echo "→ Downloading BiRefNet model..."
    "$BUNDLE_VENV/bin/python3" -c "
from transformers import AutoModelForImageSegmentation
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
print('Model downloaded.')
"
fi
echo "✓ Model cache: $(du -sh "$MODEL_CACHE" | awk '{print $1}')"

# Copy model
echo "→ Bundling model..."
cp -a "$MODEL_CACHE" "$RESOURCES/models--ZhengPeng7--BiRefNet"

# Launcher script
cat > "$MACOS/$APP_NAME" << 'LAUNCHER'
#!/bin/bash
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES="$CONTENTS/Resources"
VENV="$RESOURCES/venv"
export PATH="$VENV/bin:$PATH"
cd "$RESOURCES"
exec "$VENV/bin/python3" "$RESOURCES/rmbg_app.py"
LAUNCHER
chmod +x "$MACOS/$APP_NAME"

# Info.plist
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>AI Background Remover</string>
    <key>CFBundleIdentifier</key><string>com.andrushkevich.ai-bg-remover</string>
    <key>CFBundleVersion</key><string>$APP_VERSION</string>
    <key>CFBundleShortVersionString</key><string>$APP_VERSION</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSMultipleInstancesProhibited</key><true/>
</dict>
</plist>
PLIST

# Copy icon
ICON_SRC="$PROJECT_DIR/textures/icon.icns"
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$RESOURCES/icon.icns"
fi

echo -n "APPL????" > "$CONTENTS/PkgInfo"

# Remove quarantine
xattr -cr "$APP_DIR" 2>/dev/null || true

echo "✓ $APP_NAME.app: $(du -sh "$APP_DIR" | awk '{print $1}')"

# -------------------------------------------------------------------------
# 3. DMG (optional)
# -------------------------------------------------------------------------
if $CREATE_DMG; then
    DMG_NAME="${APP_NAME// /-}-${APP_VERSION}.dmg"

    if ! command -v create-dmg &>/dev/null; then
        echo "→ Installing create-dmg..."
        brew install create-dmg
    fi

    echo "→ Creating DMG..."
    rm -f "$DIST_DIR/$DMG_NAME"

    create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 120 \
        --app-drop-link 425 120 \
        "$DIST_DIR/$DMG_NAME" \
        "$APP_DIR" \
    || true

    if [ -f "$DIST_DIR/$DMG_NAME" ]; then
        echo "✓ DMG: $(du -sh "$DIST_DIR/$DMG_NAME" | awk '{print $1}')"
        shasum -a 256 "$DIST_DIR/$DMG_NAME"
    fi
fi

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo ""
echo "=== Build complete ==="
echo "App: $APP_DIR"
echo ""
echo "To run: open \"$APP_DIR\""
