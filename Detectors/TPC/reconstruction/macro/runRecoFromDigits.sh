#!/bin/bash

usage() {
  echo "Usage:"
  echo "runRecoFromDigits.sh [optional arguments]"
  echo
  echo "optional arguments:"
  echo "    -i, --inputFile=     : set output file name (default: '$inputFile')"
  echo "    -o, --outputType=    : set output type to be writte (default: '$outputType')"
  echo "    -q, --qMaxThreshold= : set qMax threshold for clusterization (default: '$qMaxThreshold')"
  echo "    -h, --help           : show this help message"
}

usageAndExit() {
  usage
  if [[ "$0" =~ runRecoFromDigits.sh ]]; then
    exit 0
  else
    return 0
  fi
}

# ===| default variable values |================================================
inputFile=tpcdigits.root
outputType=clusters,tracks
qMaxThreshold=4

# ===| parse command line options |=============================================
OPTIONS=$(getopt -l "inputFile:,outputType:,qMaxThreshold:,help" -o "i:o:q:h" -n "runRecoFromDigits.sh" -- "$@")

if [ $? != 0 ] ; then
  usageAndExit
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    --) shift; break;;
    -i|--inputFile) inputFile=$2; shift 2;;
    -o|--outputType) outputType=$2; shift 2;;
    -q|--qMaxThreshold) qMaxThreshold=$2; shift 2;;
    -h|--help) usageAndExit;;
     *) echo "Internal error!" ; exit 1 ;;
   esac
done

# ===| check for required arguments |===========================================

# ===| command building and execution |=========================================
cmd="o2-tpc-reco-workflow --infile $inputFile --disable-mc --configKeyValues 'TPCHwClusterer.peakChargeThreshold=$qMaxThreshold;TPCHwClusterer.isContinuousReadout=0' --output-type $outputType"
echo $cmd
eval $cmd
