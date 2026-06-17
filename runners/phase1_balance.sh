#!/usr/bin/env bash
# Phase 1: all linear backbones on natural vs balanced train (VM).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
python3 runners/phase1_balance.py "$@"
