#!/usr/bin/env bash
set -eu

clear

echo "Preparing for train streamer"
ml compileimages prepare

echo "Starting train streamer in separate process"
ml compileimages train_service &

echo "Starting python training in separate process"
../main_yolo.py __file__ &
