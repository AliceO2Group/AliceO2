o2_dir=$1

macro_dir=${o2_dir}/macro

rm -f digits.txt clusters.txt

root.exe -b -q ${macro_dir}/run_sim_tpc.C
root.exe -b -q ${macro_dir}/run_digi_tpc.C
mv AliceO2_TGeant3.tpc.digi_10_event.root o2dig.root
mv AliceO2_TGeant3.tpc.params_10.root o2sim_par.root
root.exe -b -q ${macro_dir}/run_clus_tpc.C
