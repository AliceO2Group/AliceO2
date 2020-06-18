#!/bin/bash

# Script verifying pileup / embedding features of digitizers
# It is a basic and fully automatic status check of what should
# work

# The approach is the following:
# 1) for each detector we simulate event sequences that leave (identical) hits
# 2) we digitize the events with the same collision time assigned to both of them
#    and check whether the output digits have multiple labels --> checks pileup
# 3) we digitize with trivial embedding: background and signal events are the same
#    and check whether the output digits have multiple labels --> checks embedding

dets=(         ITS   TPC     TOF    EMC    HMP      MCH     MID    MFT     FV0     FT0   FDD   TRD     PHS    CPV    ZDC )
generators=( boxgen boxgen boxgen boxgen hmpidgun fwpigen fwpigen fwpigen fddgen fddgen fddgen boxgen boxgen boxgen zdcgen )

simtask() {
 d=$1  # detector
 gen=$2 # generator

 # we execute the simulation in different directiories to achieve isolation and race-conditions
 [[ ! -d "$d" ]] && mkdir ${d}
 cd ${d}

 # we put the detector plus pipe and magnet as materials
 o2-sim -j 1 -m PIPE MAG ${d} -g ${gen} -n 1 --configKeyValues "BoxGun.number=300" --seed 1 -o o2sim${d} > simlog${d} 2>&1

 # we duplicate the events/hits a few times in order to have a sufficient
 # condition for pileup
 f=4
 origin="o2sim${d}"
 target="o2sim${d}_${f}"
 root -q -b -l ${O2_ROOT}/share/macro/duplicateHits.C\(\"${origin}\",\"${target}\",${f}\)

 # need to link the geometryfile
 ln -s "${origin}_geometry.root" "${target}_geometry.root"

 # verify that we have hits
 NUMHITS=`root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C\(\"$target\"\) | grep "${d}" | awk '//{print $2}'`
 MSG="Found ${NUMHITS} hits for ${d}"
 echo $MSG

 # digitize with extreme bunch crossing as well as with embedding the signal onto itself
 o2-sim-digitizer-workflow --onlyDet ${d} --interactionRate 1e9 -b --tpc-lanes 1 --sims ${target},${target} > digilog${d} 2>&1
}

checktask() {
  d=$1

  # find newly created digitfile
  digitfile=`ls -lt *digi*.root | head -n 1 | awk '//{print $9}'`

  root -q -b -l ${O2_ROOT}/share/macro/analyzeDigitLabels.C\(\"${digitfile}\",\"${d}\"\)
}

if [ "$(uname)" == "Darwin" ]; then
  CORESPERSOCKET=`system_profiler SPHardwareDataType | grep "Total Number of Cores:" | awk '{print $5}'`
  SOCKETS=`system_profiler SPHardwareDataType | grep "Number of Processors:" | awk '{print $4}'`
else
  # Do something under GNU/Linux platform
  CORESPERSOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
  SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
fi
N=`bc <<< "${CORESPERSOCKET}*${SOCKETS}"`
echo "Detected ${N} CPU cores"

# parallel part
for idx in "${!dets[@]}"; do

  # parallelize in bunch of N
  ((i=i%N)); ((i++==0)) && wait

  d=${dets[$idx]}
  gen=${generators[$idx]}

  simtask $d $gen &
done
wait

# checkpart
for idx in "${!dets[@]}"; do

  d=${dets[$idx]}

  cd ${d}
  checktask $d
  cd ..
done
