# Script verifying pileup / embedding features of digitizers
# It is a basic and fully automatic status check of what should
# work

# The approach is the following:
# 1) for each detector we simulate 2 events that both leave hits
# 2) we digitize the events with the same collision time assigned to both of them
#    and check whether the output digits have multiple labels --> checks pileup
# 3) we digitize with trivial embedding: background and signal events are the same
#    and check whether the output digits have multiple labels --> checks embedding

# o2-sim -j 4 -g pythia8 -n 4 --seed 1 ok for 
# CVP, MCH, MID, HMP need special gun

dets=(       ITS     TPC   TRD    EMC     PHS    TOF      CPV    HMP     MCH   MID   MFT   FV0   FT0   FDD   ZDC )
generators=(boxgen boxgen boxgen boxgen hmpidgun  boxgen  boxgen  hmpidgun fwpigen fwpigen fwpigen  fddgen fddgen fddgen pythia8 )

for idx in "${!dets[@]}"; do
  d=${dets[$idx]}
  gen=${generators[$idx]}
  # we put the detector plus pipe and magnet as materials
  o2-sim-serial -m PIPE MAG ${d} -g ${gen} -n 2 --seed 1 -o o2sim${d} > simlog${d} 2>&1

  unlink o2sim.root
  unlink o2sim_grp.root
  ln -s o2sim${d}.root o2sim.root
  ln -s o2sim${d}_grp.root o2sim_grp.root

  # verify that we have hits
  NUMHITS=`root -q -b -l ~/alisw_new/O2/macro/analyzeHits.C | grep "${d}" | awk '//{print $2}'`
  echo "Found ${NUMHITS} hits for ${d}" 
  
  # digitize with extreme bunch crossing
  o2-sim-digitizer-workflow --onlyDet ${d} -b --tpc-lanes 1 > digilog${d} 2>&1

  # verify that digit file is here
  if [ -a ${d}digits.root ]; then
    # verify that we have a single digit (here checked via the labels)
    # root -q -b -l checkEmbeddingPileupFeature("DETECTOR", file);
    echo "checking pileup for $d"

      
    # digitize with embedding but normal bunch crossing
    # o2-sim-digitizer-workflow --onlyDet ${d} -b --simFile o2sim.root --simFileS o2sim.root

    # root -q -b -l checkEmbeddingPileupFeature("DETECTOR", file);
  else
    # no digit file file
    echo "NO DIGIT FILE FOUND FOR ${d}"
  fi 
done
