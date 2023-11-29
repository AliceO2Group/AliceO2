# INPUT FILES FOR MCH CLUSTER MAPS
 INPUT FILES:
 `o2sim_geometry-aligned.root`
 `MC_QC.root`
`DATA_QC.root`



## Aligned Geometry File

- Simulation to obtain the alignement of the muon spectrometer:

```shell

o2-sim-serial --timestamp 1663632000000 -n 10 -g fwmugen -m HALL MAG DIPO COMP PIPE ABSO SHIL MCH MID

```
- The output file produced: `o2sim_geometry-aligned.root`


## MC file
SIMULATIONS:
- Go to lxplus : `ssh -X youlogin@lxplus.cern.ch`
- Source the environment : `source /cvmfs/alice-nightlies.cern.ch/bin/alienv enter VO_ALICE@O2sim::v20230413-1`
- Create a new repository for your simulation
- Check for the generator file : `O2DPG_ROOT/MC/config/PWGDQ/external/generator/GeneratorParamPromptJpsiToMuonEvtGen_pp13TeV.C` **choose the right one**

- Run the command inside the simulation repository :
```shell

  o2-sim --timestamp 1669594219618 -j 4 -n 5000 -g external -m HALL MAG DIPO COMP PIPE ABSO SHIL MCH MID -o sgn  --configKeyValues "GeneratorExternal.fileName=$O2DPG_ROOT/MC/config/PWGDQ/external/generator/GeneratorParamPromptJpsiToMuonEvtGen_pp13TeV.C;GeneratorExternal.funcName=GeneratorParamPromptJpsiToMuonEvtGen_pp13TeV()"

```

DIGITS (this can be done locally with O2 Enviroment):

```shell

 o2-sim-digitizer-workflow -b --sims=sgn

```


MC RECONSTRUCTION (this can be done locally  with QC Environment):

```shell

o2-mch-reco-workflow -b | o2-qc --config json://./qc-mch-clusters.json --local-batch=MC_QC.root | o2-dpl-run -b

```


- The output file produced: `MC_QC.root`


## Data File

DATA RECONSTRUCTION (this can be done locally  with QC Environment):

```shell

o2-ctf-reader-workflow --ctf-input /Volumes/LaData/alice/data/2022/LHC22t/529691/compact -max-tf 10 --onlyDet MCH \ | o2-mch-reco-workflow --disable-mc --disable-root-input \ | o2-qc --config json://./mch-clustermap.json --local batch="DATA_QC.root"


```

- The output file produced: `DATA_QC.root`
