#!/usr/bin/env bash
set -eu

if [ "$HOME" != "/root" ]; then echo "This script is to be run on the remote machine"; exit 1; fi

echo "Creating symbolic link for .bash_profile"

if ! [ -f ~/.bash_profile ]; then ln -s ~/remotes/install/bash_profile ~/.bash_profile; fi
if ! [ -f ~/.inputrc ]; then ln -s ~/remotes/install/inputrc ~/.inputrc; fi

