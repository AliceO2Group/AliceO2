<!-- doxy
\page refDetectorsTRDsimulation simulation
/doxy -->

# TRD simulation cheat-sheet
## Using configurable params to pass additional options to the generator for generating specific events
- Generate events with an electron particle gun in |eta| < 0.84:
```
o2-sim -m PIPE MAG TRD -n 1000 -g boxgen --configKeyValues 'BoxGun.pdg=11;BoxGun.eta[0]=-0.84;BoxGun.eta[1]=0.84'
```
- Generate events with an electron particle gun with 0.5 < p < 2.5 GeV and |eta| < 0.84:
```
o2-sim -m PIPE MAG TRD -n 1000 -g boxgen --configKeyValues 'BoxGun.pdg=11;BoxGun.eta[0]=-0.84;BoxGun.eta[1]=0.84;BoxGun.prange[0]=0.5;BoxGun.prange[1]=2.5'
```
- Generate events with a pion particle gun in |eta| < 0.84:
```
o2-sim -m PIPE MAG TRD -n 1000 -g boxgen --configKeyValues 'BoxGun.pdg=11;BoxGun.eta[0]=-0.84;BoxGun.eta[1]=0.84'
```
- Generate events with a pion particle gun with p = 2 GeV and |eta| < 0.84:
```
o2-sim -m PIPE MAG TRD -n 1000 -g boxgen --configKeyValues 'BoxGun.pdg=211;BoxGun.eta[0]=-0.84;BoxGun.eta[1]=0.84;BoxGun.prange[0]=2;BoxGun.prange[1]=2'
```
## Using TRD specific configurable params
```
--configKeyValues 'TRDSimParams.doTR=false;TRDSimParams.maxMCStepSize=0.1'
```

# Generate Raw data from MonteCarlo:

```
o2-trd-trap2raw
```
This will convert the tracklets and digits in the current directory to a series of files containing the raw data as it would appear coming out of the cru.
There are multiple options :
- -d [ --input-file-digits ] default of trddigits.root  
                                        input Trapsim digits file, empty string to have no digits.
- -t [ --input-file-tracklets ] default of trdtracklets.root
                                        input Trapsim tracklets file
-   -l [ --fileper ] how to distrbute the data into raw files. 
	- all : 1 raw file 
	- halfcru : 1 file per cru end point, so 2 files per cru. 
	- cru : one file per cru 
    - sm: one file per supermodule 
-  -o [ --output-dir ]  output directory for raw data defaults to local directory
-  -x [ --trackletHCHeader ] include tracklet half chamber header (for run3, and not in run2) 
-  -e [ --no-empty-hbf ] do not create empty HBF pages (except for HBF starting TF)
-  -r [ --rdh-version ] rdh version in use default of 6
-  --configKeyValues arg                 comma-separated configKeyValues

default would then be the following :
```
o2-trd-trap2raw -d trddigits.root -t trdtracklets.root -l halfcru -o ./ -x -r 6 -e
```
