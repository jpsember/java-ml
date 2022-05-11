#!/usr/bin/env bash
set -eu
set -x

DRYRUN=
#DRYRUN="--dry-run"


rsync --verbose --archive --recursive --exclude-from rsync_excludes.txt \
 $DRYRUN \
 . \
   -e "ssh -p${REMOTE_PORT}" \
   "eio@${REMOTE_URL}:workdir/ml"

