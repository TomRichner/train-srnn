#!/usr/bin/env bash
# Build a GCP VM image with PyTorch pre-installed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.env"

IMAGE_NAME="$GCP_IMAGE-$(date +%Y%m%d)"
TEMP_VM="image-builder-$$"
ZONE="$GCP_ZONE"

echo "Building image $IMAGE_NAME..."

# Create temporary VM
gcloud compute instances create "$TEMP_VM" \
    --project="$GCP_PROJECT" \
    --zone="$ZONE" \
    --machine-type="e2-standard-4" \
    --image-family="debian-12" \
    --image-project="debian-cloud" \
    --boot-disk-size="15GB" \
    --scopes=storage-full \
    --quiet

# Wait for SSH
sleep 30

# Install Python + PyTorch
gcloud compute ssh "$TEMP_VM" --zone="$ZONE" --project="$GCP_PROJECT" --command="
    sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git
    sudo python3 -m venv /opt/python-venv
    sudo /opt/python-venv/bin/pip install --upgrade pip
    sudo /opt/python-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    sudo /opt/python-venv/bin/pip install hydra-core omegaconf numpy pandas scipy tqdm h5py
"

# Stop VM and create image
gcloud compute instances stop "$TEMP_VM" --zone="$ZONE" --project="$GCP_PROJECT" --quiet
gcloud compute images create "$IMAGE_NAME" \
    --project="$GCP_PROJECT" \
    --source-disk="$TEMP_VM" \
    --source-disk-zone="$ZONE" \
    --family="$GCP_IMAGE_FAMILY" \
    --quiet

# Cleanup
gcloud compute instances delete "$TEMP_VM" --zone="$ZONE" --project="$GCP_PROJECT" --quiet

echo "Image $IMAGE_NAME created in family $GCP_IMAGE_FAMILY"
