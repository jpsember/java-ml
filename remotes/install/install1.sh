#!/usr/bin/env bash
set -eu


function file_dir_or_link_exists {
  test -L "$1" || test -e "$1" || test -d "$1"
}


function create_link {
  if ! file_dir_or_link_exists "$2"; then
    echo "Creating link from $1 --> $2"
    ln -s "$1" "$2"
  fi
}


if [ "$HOME" != "/root" ]; then echo "This script is to be run on the remote machine"; exit 1; fi

create_link ~/remotes/install/inputrc ~/.inputrc
create_link ~/remotes/install/bash_profile ~/.bash_profile
