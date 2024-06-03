#!/bin/bash

usage() {
  usage="Usage:
preparePedestalFiles.sh <required arguments> [optional arguments]

required arguments
-i, --inputFile=    :  input file name

optional arguments:
-o, --outputDir=          : set output directory for (default: ./)
-m, --minADC=             : minimal ADC value accepted for threshold, either one value for all ROC types,  or 'IROC,OROC' or 'IROC,OROC1,OROC2,OROC3'  (default: $minADC)
-s, --sigmaNoise=         : number of sigmas for the threshold, either one value for all ROC types,  or 'IROC,OROC' or 'IROC,OROC1,OROC2,OROC3' (default: $sigmaNoise)
-p, --pedestalOffset=     : pedestal offset value
-f, --onlyFilled          : only write links which have data
-k, --noMaskZero          : don't set pedetal value of missing pads to 1023
-h, --help                : show this help message
-n, --noisyThreshold      : threshold for noisy channel treatment (default: $noisyThreshold)
-y, --sigmaNoiseNoisy     : sigmaNoise for noisy channels (default: $sigmaNoiseNoisy)
-b, --badChannelThreshold : noise threshold to mask channels (default: $badChannelThreshold)"

  echo "$usage"
}

usageAndExit() {
  usage
  if [[ "$0" =~ preparePedestalFiles.sh ]]; then
    exit 0
  else
    return 0
  fi
}

# ===| default variable values |================================================
fileInfo=
outputDir="./"
minADC="2"
sigmaNoise="3"
pedestalOffset=0
onlyFilled=0
maskZero=1
noisyThreshold=1.5
sigmaNoiseNoisy=4
badChannelThreshold=6

# ===| parse command line options |=============================================
OPTIONS=$(getopt -l "inputFile:,outputDir:,minADC:,sigmaNoise:,pedestalOffset:,noisyThreshold:,sigmaNoiseNoisy:,badChannelThreshold:,onlyFilled,noMaskZero,help" -o "i:o:t:m:s:p:n:y:b:fkh" -n "preparePedestalFiles.sh" -- "$@")

if [ $? != 0 ] ; then
  usageAndExit
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    --) shift; break;;
    -i|--inputFile) inputFile=$2; shift 2;;
    -o|--outputDir) outputDir=$2; shift 2;;
    -m|--minADC) minADC=$2; shift 2;;
    -s|--sigmaNoise) sigmaNoise=$2; shift 2;;
    -p|--pedestalOffset) pedestalOffset=$2; shift 2;;
    -f|--onlyFilled) onlyFilled=1; shift;;
    -k|--noMaskZero) maskZero=0; shift;;
    -n|--noisyThreshold) noisyThreshold=$2; shift 2;;
    -y|--sigmaNoiseNoisy) sigmaNoiseNoisy=$2; shift 2;;
    -b|--badChannelThreshold) badChannelThreshold=$2; shift 2;;
    -h|--help) usageAndExit;;
     *) echo "Internal error!" ; exit 1 ;;
   esac
done

# ===| check for required arguments |===========================================
if [[ -z "$inputFile" ]]; then
  usageAndExit
fi

# ===| command building and execution |=========================================
cmd="root.exe -b -q -l -n -x $O2_SRC/Detectors/TPC/calibration/macro/preparePedestalFiles.C+g'(\"$inputFile\",\"$outputDir\", {$sigmaNoise}, {$minADC}, $pedestalOffset, $onlyFilled, $maskZero, $noisyThreshold, $sigmaNoiseNoisy, $badChannelThreshold)'"
echo "running: $cmd"
eval $cmd
