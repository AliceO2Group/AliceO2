#!/usr/bin/env bash
#
# This is a simple simulation example showing how to run event simulation
# using the Hijing event generator interface from AliRoot
#
# The configuration of AliGenHijing is performed by the function
# 'hijing(double energy = 5020., double bMin = 0., double bMax = 20.)' defined 
# in the macro 'aliroot_hijing.macro'.
#
# The macro file is specified via '--configKeyValues' setting 'GeneratorExternal.fileName'
# whereas the specific function call to retrieven the configuration
# is specified via 'GeneratorExternal.funcName'
# 
# IMPORTANT
# To run this example you need to load an AliRoot package compatible with the O2.
# for more details, https://alice.its.cern.ch/jira/browse/AOGM-246
#
# AliRoot needs to be loaded **after** O2 in the following sense:
# `alienv enter O2/latest,AliRoot/latest`
#

set -x

NEV=5
ENERGY=5020.
BMIN=0
BMAX=20.
o2-sim -j 20 -n ${NEV} -g external -m PIPE ITS TPC -o sim \
       --configKeyValues "GeneratorExternal.fileName=aliroot_hijing.macro;GeneratorExternal.funcName=hijing(${ENERGY}, ${BMIN}, ${BMAX})"
