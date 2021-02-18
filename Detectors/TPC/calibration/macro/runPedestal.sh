#!/bin/bash

usage() {
  echo "Usage:"
  echo "runPedestal <required arguments> [optional arguments]"
  echo
  echo
  echo "required arguments"
  echo "     -i, --fileInfo=     :  wildcard of input files in single quotes."
  echo "                            Can obtain the number of time bins separated by a ':'"
  echo "                            e.g. 'cru*.raw:1000' for 1000 time bins"
  echo "                            the time bin option after the ':' overwrites the --timeBins option"
  echo
  echo "optional arguments:"
  echo "    -o, --outputFile=    : set output file name (default: pedestals.root)"
  echo "    -f, --fistTimeBin=   : first time bin for pulser search"
  echo "    -l, --lastTimeBin=   : last time bin for pulser search"
  echo "    -n, --nevents=       : number of events to process (default: 1000)"
  echo "    -m, --adcMin=        : minimal ADC values accepted (default: 0)"
  echo "    -x, --adcMax=        : maximal ADC values accepted (default: 1100)"
  echo "    -s, --statType=      : statistics type - 0: Gaus fit (default), 1: Gaus fit fast, 2: Mean and StdDev"
  echo "    -v, --verbosity=     : set verbosity level for raw reader"
  echo "    -d, --debugLevel=    : set debug level for raw reader"
  echo "    -w, --writeDebug     : write debug output histograms"
  echo "    -h, --help           : show this help message"
}

usageAndExit() {
  usage
  if [[ "$0" =~ runPedestal.sh ]]; then
    exit 0
  else
    return 0
  fi
}

# ===| default variable values |================================================
fileInfo=
outputFile=pedestals.root
nevents=1000
firstTimeBin=0
lastTimeBin=450
statisticsType=0
verbosity=0
debugLevel=0
writeDebug=0

adcMin=0
adcMax=1100

# ===| parse command line options |=============================================
OPTIONS=$(getopt -l "fileInfo:,outputFile:,firstTimeBin:,lastTimeBin:,nevents:,adcMin:,adcMax:,statType:,verbosity:,debugLevel:,writeDebug,help" -o "i:o:t:f:l:n:m:x:s:v:d:wh" -n "runPedestal.sh" -- "$@")

if [ $? != 0 ] ; then
  usageAndExit
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    --) shift; break;;
    -i|--fileInfo) fileInfo=$2; shift 2;;
    -o|--outputFile) outputFile=$2; shift 2;;
    -f|--firstTimeBin) firstTimeBin=$2; shift 2;;
    -l|--lastTimeBin) lastTimeBin=$2; shift 2;;
    -n|--nevents) nevents=$2; shift 2;;
    -m|--adcMin) adcMin=$2; shift 2;;
    -x|--adcMax) adcMax=$2; shift 2;;
    -s|--statType) statisticsType=$2; shift 2;;
    -v|--verbosity) verbosity=$2; shift 2;;
    -d|--debugLevel) debugLevel=$2; shift 2;;
    -w|--writeDebug) writeDebug=1; shift;;
    -h|--help) usageAndExit;;
     *) echo "Internal error!" ; exit 1 ;;
   esac
done

# ===| check for required arguments |===========================================
if [[ -z "$fileInfo" ]]; then
  usageAndExit
fi

# ===| check time bin info |====================================================
if [[ $fileInfo =~ : ]]; then
  lastTimeBin=${fileInfo#*:}
  lastTimeBin=${lastTimeBin%%:*}
else
  fileInfo=${fileInfo}:${lastTimeBin}
fi

# ===| properly format fileInfo |===============================================
fileInfo=$(echo $fileInfo | sed "s|^|{\"|;s|,|:$lastTimeBin\",\"|g;s|$|\"}|")

# ===| command building and execution |=========================================
cmd="root.exe -b -q -l -n -x $O2_SRC/Detectors/TPC/calibration/macro/runPedestal.C'($fileInfo,\"$outputFile\", $nevents, $adcMin, $adcMax, $firstTimeBin, $lastTimeBin, $statisticsType, $verbosity, $debugLevel, 0, $writeDebug)'"
#cmd="perf record -g -o perf.log root.exe -b -q -l -n -x $O2_SRC/Detectors/TPC/calibration/macro/runPedestal.C'($fileInfo,\"$outputFile\", $nevents, $adcMin, $adcMax, $firstTimeBin, $lastTimeBin, $statisticsType, $verbosity, $debugLevel)'"
#cmd="valgrind --tool=callgrind --dump-instr=yes --dump-instr=yes root.exe -b -q -l -n -x $O2_SRC/Detectors/TPC/calibration/macro/runPedestal.C'($fileInfo,\"$outputFile\", $nevents, $adcMin, $adcMax, $firstTimeBin, $lastTimeBin, $statisticsType, $verbosity, $debugLevel)'"
echo "running: $cmd"
eval $cmd
