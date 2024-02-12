#!/usr/bin/env bash
set -eu

clear

echo "Perform python inference"
../driver.py inference project classifier

echo "Interpreting results"
ml compileimages oper process_inference_result
