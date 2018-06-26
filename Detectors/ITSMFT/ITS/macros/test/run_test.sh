#Number of events to simulate 
nEvents=10 

#To activate the continuos readout, assign a positive value to the rate
rate=0. #50.e3(Hz)

#MC transport
mcEngine=TGeant3

#MC event generator
mcGener=boxgen

o2sim -n $nEvents -e $mcEngine -g $mcGener -m PIPE ITS >& sim_its.log

root -b -q $O2_ROOT/share/macro/run_digi_its.C+\($rate\) >& digi_its.log

root.exe -b -q CheckDigits.C+ >& CheckDigits.log

root -b -q $O2_ROOT/share/macro/run_clus_its.C+ >& clus_its.log

root.exe -b -q CheckClusters.C+ >& CheckClusters.log

root.exe -b -q CheckTopologies.C+ >& CheckTopologies.log

root.exe -b -q $O2_ROOT/share/macro/run_trac_its.C+\($rate\) >& trac_its.log

root.exe -b -q CheckTracks.C+ >& CheckTracks.log

root.exe DisplayTrack.C+

