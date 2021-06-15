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

# Some technical details:
## Pileup implementation
There are two general cases:
**Case 1:** The first signal triggers the readout at `t=A`. The signal is read out for 30 time bins (3 microseconds). A second signal can arrive at `t=C`. The second signal will contribute to the tail of the first signal with `(B-C)*samplingRate` bins from its head.
```
A        B
**********
      **********
      C        D
```
**Case 2:** A trigger happened in the past at `t<A`, but `A` is within the deadtime of the TRD so the detector can't trigger again and the signal isn't readout. Still, the signal can contribute to a future signal. When a new signal arrives at `t=C` that can trigger, the signal will be readout for 30 time bins and will have `(B-C)*samplingRate` bins from the tail of the previous signal onto it's head.
```
A        B
**********
      **********
      C        D
```
We keep a deque of signals that is flushed when the detector gets a new trigger. All signals are stored in the deque and are pop from the front only when they are too old. A signal can contribute to two events, one in the past and one in the future. See for example **Case 3**:
```
A        B
**********
        **********
        C        D
               **********
               E        F
```
In this example, we consider a trigger at `t=A`. The second signal contributes with `(B-C)*samplingRate` bins from its head onto the tail of the first signal. A third signal arrives at `t=E`, with `E>A+BUSY_TIME`, which can trigger the detector and be readout. The third signal will have `(D-E)*samplingRate` from the tail of the second signal onto its head. So, the requirement for a signal to be too old, and be dropped, is:
- new trigger arrives, and
- the time difference between the first time bin of the previous signal and the new trigger is greater than `READOUT_TIME`.
