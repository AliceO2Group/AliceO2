#
nEvents=10
mcEngine=\"TGeant3\"
#To de-activate the ALPIDE response, set alp=kFALSE
alp=kTRUE
#To activate the continuos readout, assign a positive value to the rate 
rate=0. # 50.e3 (Hz)
root.exe -b -q $O2_ROOT/share/macro/run_sim_its_ALP3.C+\($nEvents,$mcEngine\) >& sim.log
root.exe -b -q $O2_ROOT/share/macro/run_digi_its.C+\($nEvents,$mcEngine,$alp,$rate\) >& digi.log
root.exe -b -q CheckDigits.C+\($nEvents,$mcEngine\) >& CheckDigits.log
root.exe -b -q $O2_ROOT/share/macro/run_clus_its.C+\($nEvents,$mcEngine\) >& clus.log
root.exe -b -q CheckClusters.C+\($nEvents,$mcEngine\) >& CheckClusters.log
root.exe -b -q $O2_ROOT/share/macro/run_trac_its.C+\($nEvents,$mcEngine,$rate\) >& trac.log
root.exe -b -q CheckTracks.C+\($nEvents,$mcEngine\) >& CheckTracks.log
root.exe DisplayTrack.C+\($nEvents,$mcEngine\)

