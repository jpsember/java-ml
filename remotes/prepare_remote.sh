#!/usr/bin/env bash
set -eu

echo "Copying install subdirectory to remote"
dev push install . -v

echo "Exec commands on remote"
sshe '(cd install; pwd; bash install.sh)'
