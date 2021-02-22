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
