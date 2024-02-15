#!/usr/bin/env bash
set -eu

echo "Compiling training images"
ml genimages

echo "Compiling evaluation images"
ml compileimages oper compile_inference_images

