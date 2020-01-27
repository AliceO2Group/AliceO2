#!/bin/bash
set -x

# Script verifying pileup / embedding features of digitizers
# It is a basic and fully automatic status check of what should
# work

# The approach is the following:
# 1) for each detector we simulate event sequences that leave (identical) hits
# 2) we digitize the events with the same collision time assigned to both of them
#    and check whether the output digits have multiple labels --> checks pileup
# 3) we digitize with trivial embedding: background and signal events are the same
#    and check whether the output digits have multiple labels --> checks embedding

# let's do a global simulation with a generator + seed combination supposed to leave hits
# in all detectors (for now pythia8 but this we might have to change)

o2-sim -g pythia8hi --seed 1 -n 1 -j 20 -o o2simglobal > simlog_global 2>&1

# we duplicate the events/hits a few times in order to have a sufficient
# condition for pileup
f=4
origin="o2simglobal.root"
target="o2simglobal_${f}.root"
root -q -b -l ${O2_ROOT}/share/macro/duplicateHits.C\(\"${origin}\",\"${target}\",${f}\)

unlink o2sim.root
unlink o2sim_grp.root
ln -s ${target} o2sim.root
ln -s o2simglobal_grp.root o2sim_grp.root

root -q -b -l ${O2_ROOT}/share/macro/analyzeHits.C > hitcounts.txt

dets=(  ITS  TPC  TOF  EMC  HMP  MCH  MID  MFT  FV0  FT0  FDD  TRD  PHS CPV  ZDC )

hitfile="dethits.txt"
labelfile="detlabels.txt"
for idx in "${!dets[@]}"; do
  d=${dets[$idx]}

  # verify that we have hits
  NUMHITS=`grep "${d}" hitcounts.txt | awk '//{print $2}'`
  MSG="Found ${NUMHITS} hits for ${d}"
  echo $MSG >> $hitfile

  # digitize with extreme bunch crossing as well as with embedding the signal onto itself
  o2-sim-digitizer-workflow --onlyDet ${d} --interactionRate 1e9 -b --simFileS o2sim.root > digilog${d} 2>&1

  # find newly created digitfile
  digitfile=`ls -lt *digi*.root | head -n 1 | awk '//{print $9}'`

  labeloutfile="labellog_${d}.log"
  root -q -b -l ${O2_ROOT}/share/macro/analyzeDigitLabels.C\(\"${digitfile}\",\"${d}\"\) | grep "embedding" > ${labeloutfile} 2>&1
  cat $labeloutfile >> ${labelfile}
done

paste ${hitfile} ${labelfile} > check_summary.txt
