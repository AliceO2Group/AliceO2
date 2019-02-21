#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2
buildDir="build/"

rsyncBlacklist="--exclude=$buildDir \
                --exclude=.git/ \
                --exclude=measurements \
                --exclude=*.bin \
                --exclude=*.swp \
                --exclude=*.BACK \
                --exclude=tags" 

rsyncFlags="-aPEh $rsyncBlacklist"
toSync="."

rsync $rsyncFlags $toSync $remoteTgt:$tgtDir

