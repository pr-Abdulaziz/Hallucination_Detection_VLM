#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VG_DIR="${REPO_ROOT}/vg/images"
mkdir -p "${VG_DIR}"

echo "Starting Visual Genome dataset download..."

# Download parts if they don't exist
if [ ! -f "images.zip" ]; then
  echo "Downloading images.zip (~9GB)..."
  wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
fi

if [ ! -f "images2.zip" ]; then
  echo "Downloading images2.zip (~5GB)..."
  wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
fi

echo "Extracting images.zip..."
unzip -q -j images.zip -d "${VG_DIR}"

echo "Extracting images2.zip..."
unzip -q -j images2.zip -d "${VG_DIR}"

echo "Cleaning up ZIP files..."
rm images.zip images2.zip

echo "VG installation completed."
ls -lh "${VG_DIR}" | head -n 10
