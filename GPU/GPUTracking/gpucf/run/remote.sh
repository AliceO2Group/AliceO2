#!/bin/bash

set -euo pipefail

remoteTgt=$1
tgtDir=$2
buildDir="build/"

rsyncBlacklist="--exclude=$buildDir \
                --exclude=.git/ \
                --exclude=*.bin \
                --exclude=*.swp \
                --exclude=tags" 

rsyncFlags="-aPEh $rsyncBlacklist"
toSync="."

rsync $rsyncFlags $toSync $remoteTgt:$tgtDir

ssh $remoteTgt "cd $tgtDir; [ -d $buildDir ] || (mkdir $buildDir && cd $buildDir && cmake ..)"
