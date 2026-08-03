#!/usr/bin/env bash

# A simple example showing how to merge several event pools.
# The pools are listed in pools.txt. In the example they are pulled from AliEn, so an alien token
# is needed and the files are copied locally before merging.
# Additionally this could be run as
#   o2-generators-merge-evtpool -i poolA.root,poolB.root -o merged.root
# for local files, or
#   o2-generators-merge-evtpool -i pools.txt,alien:///alice/cern.ch/user/p/pwgpp/Test/pools.txt
# for a mix of local and AliEn files.

set -x

# merge the pools listed in pools.txt, downloading 2 of them in parallel
o2-generators-merge-evtpool -i pools.txt -o evtpool.root -j 2