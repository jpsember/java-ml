#!/usr/bin/env bash
set -eu

if [ "$HOME" != "/root" ]; then echo "This script is to be run on the remote machine"; exit 1; fi

echo "Creating symbolic link for .bash_profile"

ln -s ~/remotes/linode/bash_profile ~/.bash_profile
ln -s ~/remotes/linode/inputrc ~/.inputrc

apt-get update

