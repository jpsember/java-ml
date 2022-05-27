#!/usr/bin/env bash
set -eu

datagen clean
datagen --args datagen-args-python.json clean
