#!/bin/sh -ex
# This is to create a dummy entry to CCDB.

ORIGIN=${ORIGIN:-TST}
DESCRIPTION=${DESCRIPTION:-FOO}
URL=${URL:-localhost}

START_IOV=`echo "$(date +%s) *1000" | bc`
END_IOV=`echo "$(date +%s) *1000 + 86000000" | bc`

dd if=/dev/urandom count=1 bs=1024 | curl -F "blob=@-" $URL/$ORIGIN/$DESCRIPTION/$START_IOV/$END_IOV
