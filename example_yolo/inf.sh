#!/usr/bin/env bash
set -eu

echo "Perform python inference"
../driver.py inference project yolo
