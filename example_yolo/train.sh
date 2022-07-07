#!/usr/bin/env bash
set -eu

clear

#echo "Deleting checkpoints"
#delcp.sh

touch progress.txt

#echo "Preparing for train streamer"
ml compileimages prepare

#echo "Starting train streamer in separate process"
ml compileimages train_service &

#echo "Starting python training in separate process"
../main_yolo.py __file__ &

# Watching progress file
tail -f progress.txt
