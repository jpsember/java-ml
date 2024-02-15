#!/usr/bin/env bash
set -eu

echo "Creating symbolic link for .bash_profile"

ln -s ~/remotes/linode/bash_profile ~/.bash_profile
ln -s ~/remotes/linode/inputrc ~/.inputrc
