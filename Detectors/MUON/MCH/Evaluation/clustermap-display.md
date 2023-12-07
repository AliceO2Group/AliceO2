# MCH Cluster Maps

 - Goal : The general purpose is to track "unexpected" detector issues not well reproduced with MC simulations. These problems generate non-negligible bias in Acc*Eff corrections resulting in large tracking systematic uncertainties. During the data reconstruction, the status of the detector is calculated with the CCDB which is used to discard most of the detector issues. This status map is built with information based on pedestals, occupancy etc. (high and low voltage will be included soon in the statusmap.) Nevertheless, some detector issues (e.g. a cable swapping) are not well detected online and consequently not properly reproduced by the CCBD. The main objective of this code is to spot these issues not included in the status map.

 - SRC FILE:
 `clustermap-display.cxx`

- INPUT FILES:
`DATA_QC.root`
`MC_QC.root`
`o2sim_geometry-aligned.root`

 - HELP MESSAGE TO KNOW WHICH OPTIONS ARE AVAILABLE:
```shell

o2-mch-clustermap-display --help

```

 - EXECUTION COMMAND:

```shell

o2-mch-clustermap-display --green --normperarea --rootfileleft DATA_QC.root --rootfileright 100mil.root --help

```

  - OUTPUT FILES:

 Non-bending(NB):

 ```shell

CHAMBERS-1-NB.html CHAMBERS-2-NB.html CHAMBERS-3-NB.html CHAMBERS-4-NB.html CHAMBERS-5-NB.html CHAMBERS-6-NB.html CHAMBERS-7-NB.html CHAMBERS-8-NB.html CHAMBERS-9-NB.html CHAMBERS-10-NB.html

```
Bending(B):
 ```shell

CHAMBERS-1-B.html CHAMBERS-2-B.html CHAMBERS-3-B.html  CHAMBERS-4-B.html CHAMBERS-5-B.html CHAMBERS-6-B.html CHAMBERS-7-B.html CHAMBERS-8-B.html CHAMBERS-9-B.html  CHAMBERS-10-B.html

```


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


**NOTICE THAT THE OUTPUT IS A QUALITY CONTROL OBJECT (NOT A TH1F) -- FILE CONVERTION STEP STILL NEEDED**

**For the moment, we can save the output file with the browser as a TH1F**



