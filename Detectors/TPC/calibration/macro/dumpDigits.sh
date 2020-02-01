#!/bin/bash

usage() {
cat<<EOF
Usage:
dumpDigits.sh <required arguments> [optional arguments]


required arguments
     -i, --fileInfo=     :  wildcard of input files in single quotes.
                            Can obtain the number of time bins separated by a ':'
                            e.g. 'cru*.raw:1000' for 1000 time bins
                            the time bin option after the ':' overwrites the --timeBins option
    -p, --pedestalFile=  : pedestal and noise file

optional arguments:
    -o, --outputFile=    : set output file name (default: '$outputFile')
    -f, --fistTimeBin=   : first time bin for pulser search
    -l, --lastTimeBin=   : last time bin for pulser search
    -n, --nevents=       : number of events to process (default: $nevents)
    -m, --adcMin=        : minimal ADC value accepted (default: $adcMin)
    -x, --adcMax=        : maximal ADC value accepted (default: $adcMax)
    -t, --threshold=     : threshold in nSigma noise (default: $threshold)
    -v, --verbosity=     : set verbosity level for raw reader
    -d, --debugLevel=    : set debug level for raw reader
    -h, --help           : show this help message
EOF
}

usageAndExit() {
  usage
  if [[ "$0" =~ dumpDigits.sh ]]; then
    exit 0
  else
    return 0
  fi
}

# ===| default variable values |================================================
fileInfo=
outputFile=tpcdigits.root
nevents=1000
firstTimeBin=0
lastTimeBin=475
verbosity=0
debugLevel=0
pedestalFile=""

adcMin=3
adcMax=1100
threshold=3

# ===| parse command line options |=============================================
OPTIONS=$(getopt -l "fileInfo:,outputFile:,firstTimeBin:,lastTimeBin:,nevents:,adcMin:,adcMax:,threshold:,pedestalFile:,verbosity:,debugLevel:,help" -o "i:o:f:l:n:m:x:t:p:v:d:h" -n "dumpDigits.sh" -- "$@")

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
    -t|--threshold) threshold=$2; shift 2;;
    -p|--pedestalFile) pedestalFile=$2; shift 2;;
    -v|--verbosity) verbosity=$2; shift 2;;
    -d|--debugLevel) debugLevel=$2; shift 2;;
    -h|--help) usageAndExit;;
     *) echo "Internal error!" ; exit 1 ;;
   esac
done

# ===| check for required arguments |===========================================
if [[ -z "$fileInfo" ]]||[[ -z "$pedestalFile" ]]; then
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
cmd="root.exe -b -q -l -n -x $O2_SRC/Detectors/TPC/calibration/macro/dumpDigits.C'($fileInfo,\"$outputFile\", $nevents, $adcMin, $adcMax, $firstTimeBin, $lastTimeBin, $threshold, \"$pedestalFile\", $verbosity, $debugLevel)' &> digitDump.log"
echo "running: $cmd"
eval $cmd
