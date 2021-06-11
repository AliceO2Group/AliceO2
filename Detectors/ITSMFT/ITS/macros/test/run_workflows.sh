#!/bin/bash
o2-sim -n 10 -e TGeant3 -g boxgen -m PIPE ITS >& sim.log
o2-sim-digitizer-workflow >& digi.log
#o2-sim-digitizer-workflow --configKeyValues "ITSDigitizerParam.continuous=0" >& digi.log
o2-its-reco-workflow >& reco.log
# o2-its-reco-workflow --trackerCA >& reco.log

# These macros can be run to check the quality of the results
# root.exe CheckDigits.C+
# root.exe CheckClusters.C+
# root.exe CreateDictionaries.C+
# root.exe CheckTracks.C+
# root.exe DisplayTrack.C+

