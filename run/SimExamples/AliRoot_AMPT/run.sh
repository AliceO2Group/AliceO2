#!/usr/bin/env bash
#
# Please, refer to the README.md for more information
#

set -x

NEV=5
ENERGY=5020.
BMIN=15.
BMAX=20.
o2-sim -j 20 -n ${NEV} -g external -m PIPE ITS TPC -o sim \
       --configKeyValues "GeneratorExternal.fileName=aliroot_ampt.macro;GeneratorExternal.funcName=ampt(${ENERGY}, ${BMIN}, ${BMAX})"
