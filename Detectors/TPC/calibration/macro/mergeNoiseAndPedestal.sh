#!/bin/bash

usage() {
  echo "Usage:"
  echo "mergeNoiseAndPedestal.sh [optional arguments]"
  echo
  echo
  echo "optional arguments:"
  echo "    -o, --outputFile=    : set output file name (default: mergedPedestalsAndNoise.root)"
  echo "    -i, --inputFile=     : set input file names wildcard (default: 'Event*.root')"
  echo "                           in case of a wild card it must be put in '' to not"
  echo "                           be expanded by the shell"
  echo "    -h, --help           : show this help message"
}

usageAndExit() {
  usage
  if [[ "$0" =~ mergeNoiseAndPedestal.sh ]]; then
    exit 0
  else
    return 0
  fi
}

# ===| default variable values |================================================
inputFile='Event*.root'
outputFile=mergedPedestalsAndNoise.root

# ===| parse command line options |=============================================
OPTIONS=$(getopt -l "inputFile:,outputFile:,help" -o "i:o:h" -n "mergeNoiseAndPedestal.sh" -- "$@")

if [ $? != 0 ] ; then
  usageAndExit
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    --) shift; break;;
    -i|--inputFile) inputFile=$2; shift 2;;
    -o|--outputFile) outputFile=$2; shift 2;;
    -h|--help) usageAndExit;;
     *) echo "Internal error!" ; exit 1 ;;
   esac
done

# ===| check for required arguments |===========================================
#if [[ -z "$fileInfo" ]]; then
  #usageAndExit
#fi

# ===| command building and execution |=========================================
cmd="root.exe -b -q -l -n -x $calibMacroDir/mergeNoiseAndPedestal.C'(\"$inputFile\",\"$outputFile\")'"
echo "running: $cmd"
eval $cmd
