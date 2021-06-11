#Number of events to simulate 
nEvents=10 

#MC transport
mcEngine=TGeant3

#MC event generator
mcGener=boxgen

o2-sim -n $nEvents -e $mcEngine -g $mcGener -m PIPE ITS >& sim_its.log

# Digitizing in triggered readout mode...
o2-sim-digitizer-workflow -b --configKeyValues "ITSDigitizerParam.continuous=0" >& digi.log 

root.exe -b -q CheckDigits.C+ >& CheckDigits.log

root -b -q $O2_ROOT/share/macro/run_clus_itsSA.C+\(\"itsdigits.root\",\"o2clus_its.root\",false\) >& clus_its.log

root.exe -b -q CheckClusters.C+ >& CheckClusters.log

#root.exe -b -q CreateDictionaries.C+ >& CreateDictionaries.log

root.exe -b -q $O2_ROOT/share/macro/run_trac_its.C+ >& trac_its.log

root.exe -b -q CheckTracks.C+ >& CheckTracks.log

#root.exe DisplayTrack.C+

