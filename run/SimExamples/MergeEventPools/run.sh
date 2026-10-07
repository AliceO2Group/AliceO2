#!/usr/bin/env bash

# A simple example showing how to merge several event pools.
# The pools are listed in pools.txt. In the example they are read from AliEn, so an alien token
# is needed.
# Additionally this could be run as
#   o2-generators-merge-evtpool -i poolA.root,poolB.root -o merged.root
# to give the pools directly, or
#   o2-generators-merge-evtpool -i poolA.root,pools.txt -o merged.root
# to mix single pools with a list. The list files themselves must be local.

set -x

# merge the pools listed in pools.txt
o2-generators-merge-evtpool -i pools.txt -o evtpool.root
