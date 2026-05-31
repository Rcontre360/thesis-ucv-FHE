#!/usr/bin/env bash
# Prepare a fresh Ubuntu machine to run the FHE benchmarks.
# Installs system + build deps (per README troubleshooting), Go (per docs/ORION.md),
# clones Orion, then provisions .env and .env-orion via tox.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. System packages (HEonGPU build + Orion CGO build + Python toolchain).
sudo apt-get update
sudo apt-get install -y \
    build-essential pkg-config git ninja-build wget \
    libgmp-dev libntl-dev zlib1g-dev libssl-dev \
    python3.12 python3.12-venv python3-pip pipx

# 2. Go >= 1.22 (Orion's Lattigo CGO bridge needs it).
need_go=1
if command -v go >/dev/null 2>&1; then
    GO_VER=$(go version | awk '{print $3}' | sed 's/go//')
    GO_MAJOR=$(echo "$GO_VER" | cut -d. -f1)
    GO_MINOR=$(echo "$GO_VER" | cut -d. -f2)
    if [ "$GO_MAJOR" -gt 1 ] || { [ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -ge 22 ]; }; then
        need_go=0
    fi
fi
if [ "$need_go" -eq 1 ]; then
    cd /tmp
    wget -q https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
    rm go1.22.3.linux-amd64.tar.gz
    grep -qF '/usr/local/go/bin' "$HOME/.bashrc" || echo 'export PATH=/usr/local/go/bin:$PATH' >> "$HOME/.bashrc"
    export PATH=/usr/local/go/bin:$PATH
fi

# 3. CUDA Toolkit 12.8 — checked, not auto-installed (drivers + toolkit are too
#    machine-specific to script safely).
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not on PATH. Install CUDA Toolkit 12.8 from NVIDIA before re-running." >&2
    exit 1
fi

# 4. Clone Orion source (tox -e orion installs it editable from sdk/temp/orion).
if [ ! -d "${SDK_DIR}/temp/orion" ]; then
    mkdir -p "${SDK_DIR}/temp"
    git clone https://github.com/baahl-nyu/orion.git "${SDK_DIR}/temp/orion"
fi

# 5. Bootstrap tox into the user PATH and provision both bench venvs.
pipx install tox >/dev/null 2>&1 || pipx upgrade tox >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
cd "${SDK_DIR}"
tox -e main
tox -e orion

echo "----------------------------------------"
echo "Done. To run the benchmarks:"
echo "  source ${SDK_DIR}/.env/bin/activate"
echo "  ${SDK_DIR}/scripts/benchmarks.sh playground duration"
