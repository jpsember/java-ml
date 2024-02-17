#!/usr/bin/env sh
set -eu

FILE="mk"

if [ ! -f ${FILE} ]; then

	FILE=.jsproject/make.sh
	if [ ! -f ${FILE} ]; then
		echo "Calling dev to create a make file"
		dev createmake
	fi

fi

${FILE} "$@"

# Look for an auxilliary make step

FILE="mk_aux"
if [ -f ${FILE} ]; then
	${FILE} "$@"
fi
