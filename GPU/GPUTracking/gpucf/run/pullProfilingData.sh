#!/bin/bash

set -euo pipefail

host=$1
tgtDir=$2

tmpDir='/tmp/gpucf_profiling'

scp $host:$tmpDir/* $tgtDir
