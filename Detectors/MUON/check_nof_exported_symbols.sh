#!/bin/sh
# check if number of exported symbols in library is some number

library=$1
expected=$2

test $(/usr/bin/nm -m -extern-only -defined-only $library -s __TEXT __text | wc -l) -eq $expected