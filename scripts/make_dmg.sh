#!/usr/bin/env bash
# Package dist/Phoneme.app into a shareable DMG. Run after build_mac_app.py.
set -euo pipefail

APP="dist/Phoneme.app"
DMG="dist/Phoneme.dmg"

if [ ! -d "$APP" ]; then
    echo "Phoneme.app not found at $APP — run 'uv run python scripts/build_mac_app.py py2app' first."
    exit 1
fi

rm -f "$DMG"
hdiutil create \
    -volname "Phoneme" \
    -srcfolder "$APP" \
    -ov -format UDZO \
    "$DMG"

echo
echo "Wrote $DMG"
echo "Share it. First launch: right-click → Open (Gatekeeper)."
