#!/usr/bin/env bash
set -eu

#if [ $# -ne 1 ]; then echo "illegal number of parameters; please specify a password"; exit 1; fi
#if [ "$1" == "" ]; then echo "please enter a password" ; exit 1; fi


# Read access token, strip trailing whitespace

TOKEN=`cat linode_access_token.txt`
TOKEN=`echo $TOKEN | xargs`

PWD=`cat linode_password.txt`
PWD=`echo $TOKEN | xargs`

curl -H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-X POST -d '{
    "authorized_users": [
        "jpsember"
    ],
    "backups_enabled": false,
    "booted": true,
    "image": "linode/ubuntu20.04",
    "label": "cheapcpu",
    "private_ip": false,
    "region": "us-sea",
    "root_pass": "'$PWD'",
    "tags": [],
    "type": "g6-nanode-1"
}' https://api.linode.com/v4/linode/instances
