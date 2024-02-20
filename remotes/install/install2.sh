#!/usr/bin/env bash
set -eux

if [ "$HOME" != "/root" ]; then echo "This script is only to be run on the remote machine"; exit 1; fi



function getdep {
  echo "Installing dependency from github: $1"
  rm -rf $1
  git clone https://github.com/jpsember/$1.git
  (cd $1; mk skiptest)
}





mkdir -p /root/bin


echo "Updating software"

apt-get update
apt install -y maven
apt install -y openjdk-11-jre-headless
apt install -y python3-pip
#pip install jstyleson numpy torch

echo "exiting early"
exit 0







echo "Cloning git repositories"

mkdir -p repos
cd repos

getdep "java-core"
getdep "java-testutil"
getdep "datagen"
getdep "java-webtools"
getdep "java-graphics"
getdep "dev"

# This creates ~/bin/mk, which is used by subsequent java projects
#
dev setup new
getdep "java-ml"

cd ..

echo "Done install.sh"
