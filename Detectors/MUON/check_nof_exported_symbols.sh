#!/bin/sh
# check if number of exported symbols in library is some number

library=$1
expected=$2

nlibs=$(/usr/bin/nm -m -extern-only -defined-only $library -s __TEXT __text | grep mch | wc -l)

if [ $nlibs -ne $expected ]; then
  echo "bad: check number of exported symbols in $library"
  /usr/bin/nm -m -extern-only -defined-only $library -s __TEXT __text
else
  echo "good: $library contains the expected $expected exported symbols"
fi
