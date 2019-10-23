#!/bin/bash

set -euo pipefail

CMAKE_ARGS=$@

scriptDir=$(dirname $0)
baseDir="$scriptDir/.."
buildDir="$baseDir/build"
rm -rf $buildDir

mkdir -p $buildDir/debug
pushd $buildDir/debug
cmake ../.. -DCMAKE_BUILD_TYPE=Debug $@
popd

mkdir -p $buildDir/release
cd $buildDir/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release $@
