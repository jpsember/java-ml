#!/usr/bin/env bash
set -eu

if [ "$HOME" == "/root" ]; then echo "This script is NOT to be run on the remote machine"; exit 1; fi

echo "Copying install subdirectory to remote"
dev push install . -v

echo "Exec commands on remote"
sshe 'bash --login -c install/install1.sh'
sshe 'bash --login -c install/install2.sh'

