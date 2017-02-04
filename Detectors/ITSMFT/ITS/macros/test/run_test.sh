#
root.exe -b -q run_sim.C+ >& sim.log
root.exe -b -q run_digi.C+ >& digi.log
root.exe -b -q CheckDigits.C+ >& CheckDigits.log
root.exe -b -q run_clus.C+ >& clus.log
root.exe -b -q CheckClusters.C+ >& CheckClusters.log
root.exe -b -q run_trac.C+ >& trac.log
root.exe -b -q CheckTracks.C+ >& CheckTracks.log
root.exe DisplayTrack.C+

