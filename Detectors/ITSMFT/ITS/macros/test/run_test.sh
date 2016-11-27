#
root.exe -b -q run_sim.C+ >& sim.log
root.exe -b -q run_digi.C+ >& digi.log
root.exe -b -q run_clus.C+ >& clus.log
# root.exe CheckDigits.C+
root.exe CheckClusters.C+

