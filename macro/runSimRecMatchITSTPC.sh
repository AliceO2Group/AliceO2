nEvents=10
#To activate the continuos readout, assign a positive value to the rate 
rate=50.e3

o2sim -n $nEvents -m PIPE ITS TPC FRAME >& sim.log

macroPath="$O2_ROOT/share/macro"
preload="$macroPath"/loadExtDepLib.C

simFile='"o2sim.root"'
parFile='"o2sim_par.root"'
grpFile='"o2sim_grp.root"'
digFile='"o2dig.root"';
clsITSFile='"o2clus_its.root"'
clsTPCFile='"o2clus_tpc.root"'
clsTPCHWFile='"o2clus_tpc_HW.root"'
clsTPCNatFile='"o2clus_tpc_Native.root"'
trcITSFile='"o2track_its.root"'
trcTPCFile='"o2track_tpc.root"'
matchFile='"o2match_itstpc.root"'
geomFile='"O2geometry.root"'

root -l -b -q $preload "$macroPath"/run_digi_all.C+\($rate,$digFile,$simFile,$parFile,$grpFile\) >& dig.log
root -l -b -q $preload "$macroPath"/run_clus_its.C+\($clsITSFile,$digFile,$parFile\) >& clus_its.log
root -l -b -q $preload "$macroPath"/run_clus_tpc.C+\($clsTPCFile,$digFile,$parFile\) >& clus_tpc.log

root -l -b -q $preload "$macroPath"/run_trac_its.C+\($rate,$trcITSFile,$clsITSFile,$parFile\) >& rec_its.log

#convert old clusters to ClusterHardware
root -l -b -q $preload "$macroPath"/convertClusterToClusterHardware.C+\($clsTPCFile,$clsTPCHWFile\) >& convOls2HW.log
#convert ClusterHardware to ClusterNative
root -l -b -q $preload "$macroPath"/runHardwareClusterDecoderRoot.C+\($clsTPCHWFile,$clsTPCNatFile\) >& convHV2Native.log
#Run tracking on ClusterNative type
root -l -b -q $preload "$macroPath"/runCATrackingClusterNative.C+\($clsTPCNatFile,$trcTPCFile,'"cont refX=83 bz=-5.0068597793"'\) >& tpcTracking.log

root -l -b -q $preload "$macroPath"/run_match_TPCITS.C+\(\"./\",$matchFile,$trcITSFile,$trcTPCFile,$clsITSFile,$geomFile,$grpFile\) >& matcTPCITS.log
