#
root.exe -b -q run_sim.C+ >& sim.log
root.exe -b -q run_digi.C+ >& digi.log
# root.exe CheckDigits.C+
root.exe -b -q run_clus.C+ >& clus.log
root.exe CheckClusters.C+
root.exe -b -q run_trac.C+ >& trac.log
# root.exe CheckTracks.C+

