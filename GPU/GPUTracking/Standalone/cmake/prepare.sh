#!/bin/bash
ALIARCH=`aliBuild architecture`
if [[ -z $ALIARCH ]]; then
  echo "Error obtaining aliBuild architecture"
  return 1
fi
if [[ -z $ALIBUILD_WORK_DIR ]]; then
  WORK_DIR="`pwd`/sw"
else
  WORK_DIR="$ALIBUILD_WORK_DIR"
fi
eval "`alienv shell-helper`"
alienv load O2/latest
for i in Vc boost fmt CMake ms_gsl Clang ninja; do
  source sw/$ALIARCH/$i/latest/etc/profile.d/init.sh
done
