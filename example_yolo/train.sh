#!/usr/bin/env bash
set -eu

clear

touch progress.txt

#echo "Preparing for train streamer"
ml compileimages prepare

#echo "Starting train streamer in separate process"
ml compileimages train_service &

#echo "Starting python training in separate process"
../driver.py train project yolo &

# Watching progress file
tail -f progress.txt
