#!/usr/bin/env bash
# Install teensyvad + the AudioSocket VAD server.
#   sudo ./deploy/install.sh          (Linux/Asterisk host or macOS)
set -euo pipefail
DEST=/opt/teensyvad
SRC="$(cd "$(dirname "$0")/.." && pwd)"
sudo mkdir -p "$DEST"
sudo rsync -a --exclude '.venv' --exclude 'data' --exclude '.git' "$SRC/" "$DEST/"
cd "$DEST"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q numpy huggingface_hub
sudo cp deploy/teensyvad-audiosocket.service /etc/systemd/system/ 2>/dev/null || true
sudo cp deploy/com.teensyvad.audiosocket.plist /Library/LaunchDaemons/ 2>/dev/null || true
sudo systemctl daemon-reload 2>/dev/null && sudo systemctl enable --now teensyvad-audiosocket && \
  echo "deployed: systemctl status teensyvad-audiosocket" || \
  { sudo launchctl unload /Library/LaunchDaemons/com.teensyvad.audiosocket.plist 2>/dev/null || true
    sudo launchctl load /Library/LaunchDaemons/com.teensyvad.audiosocket.plist && \
    echo "deployed: sudo launchctl list | grep teensyvad"; }
