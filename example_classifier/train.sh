#!/usr/bin/env bash
set -eu

clear

touch progress.txt

echo "Preparing for train streamer"
ml compileimages oper prepare_train

echo "Starting train streamer in separate process"
ml compileimages oper train_service &

echo "Starting python training in separate process"
../driver.py train project classifier &

# Watching progress file
tail -f progress.txt

