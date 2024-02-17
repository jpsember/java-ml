#!/usr/bin/env bash
set -eux

# Add '.' to the path in case it doesn't exist
# (not sure this will work)


mkdir -p /root/bin



if [ "$HOME" != "/root" ]; then echo "This script is only to be run on the remote machine"; exit 1; fi



getdep () {
  echo "Installing dependency from github: $1"
  rm -rf $1
  git clone https://github.com/jpsember/$1.git
  (cd $1; mk)
}


echo "Cloning git repositories"
mkdir -p repos
cd repos

if false; then
getdep "java-core"
getdep "java-testutil"
getdep "datagen"
getdep "java-webtools"
getdep "java-graphics"
fi
getdep "dev"

dev setup new

getdep "java-ml"

cd ..


echo "Installing .bash_profile, .inputrc"

cp -f bash_profile $HOME/.bash_profile
cp -f inputrc $HOME/.inputrc


echo "Updating software"

apt-get update
apt install -y maven
apt install -y openjdk-11-jre-headless

echo "Done install.sh"
