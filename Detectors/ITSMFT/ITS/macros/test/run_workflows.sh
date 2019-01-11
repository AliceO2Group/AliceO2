#!/bin/bash
o2sim -n 10 -e TGeant3 -g boxgen -m PIPE ITS >& sim.log 
digitizer-workflow >& digi.log
#digitizer-workflow --ITStriggered >& digi.log
its-reco-workflow >& reco.log

# These macros can be run to check the quality of the results 
# root.exe CheckDigits.C+
# root.exe CheckClusters.C+
# root.exe CheckTopologies.C+
# root.exe CheckTracks.C+
# root.exe DisplayTrack.C+

