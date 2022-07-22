#!/usr/bin/env bash
set -eu

echo "Perform python inference"
../driver.py inference project yolo

echo "Interpreting results"
ml compileimages oper process_inference_result
