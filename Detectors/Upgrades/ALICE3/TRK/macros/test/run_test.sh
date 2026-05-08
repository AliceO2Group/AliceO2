#Number of events to simulate
nEvents=10

# Simulating
o2-sim-serial-run5 -n $nEvents -g pythia8hi -m TRK --configKeyValues "TRKBase.layoutML=kTurboStaves;TRKBase.layoutOT=kStaggered;">& sim_TRK.log

# Digitizing
o2-sim-digitizer-workflow -b >& digiTRK.log

root.exe -b -q CheckDigits.C+ >& CheckDigits.log
