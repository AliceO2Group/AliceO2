<!-- doxy
\page refTPCcalibrationCalibdEdx Residual dEdx Calibration
/doxy -->

# Residual dEdx Calibration

The workflow `o2-tpc-miptrack-filter` will run on the EPNs. It selects only the tracks inside the defined cuts and stream them with the name `TPC/MIPS/0`. The cut values can be changed with these CLI options:

```
--min-momentum (= 0.3)
--max-momentum (= 0.7)
--min-dedx (= 20)           Minimum dEdx cut
--max-dedx (= 200)          Maximum dEdx cut
--min-clusters (= 60)       Minimum number of cluster in a track
```

The workflow `o2-tpc-calibratordedx` should run on an aggregation node. It fills dEdx histograms using the data sent by the `o2-tpc-miptrack-filter`, and computes corrections for the dE/dx values for every time slot.

```
--tf-per-slot           TFs per calibration time slot
--max-delay             Slots in past to consider
--min-entries           Minimum number of entries per GEM stack to perform a fit

--min-entries-sector    Minimum entries per GEM stack to enable sector by sector correction. Below this value we only perform one fit per ROC type (IROC, OROC1, ...; no side nor sector information)
--min-entries-1d        Minimum entries per stack to perform a 1D fit, bellow it we only calculate the mean of each gem stack
--min-entries-2d        Mininum entries per stack to perform a 2D fit
--fit-passes            Number of fit iterations
--fit-threshold         dEdx cut width around the MIP peak used in the fit

--dedxbins              Number of dE/dx bins
--angularbins           Number of angular bins, for values like Tgl and Snp
--min-dedx              Min. dE/dx value
--max-dedx              Max. dE/dx value
--fit-snp               Enable Snp correction

--file-dump             Save calibration correction to a file
--field                 Magnetic field in kG, need for track propagations, this value will be overwritten if a grp file is present
```

The workflow `o2-tpc-calibdedx` is similar to `o2-tpc-calibratordedx`, but compute only one correction using all available time frames.

## Executing

The full dE/dx calibration workflow can be executed in the following way:

```
o2-tpc-track-reader --disable-mc | o2-tpc-miptrack-filter | o2-tpc-calibrator-dedx --file-dump --min-entries 100 --tf-per-slot 10
```

Where we enabled the option to save the corrections to a file, set the min. entries per time slot to 100 and defined that the time slots should have 10 time frames.

## Simulating EPN workflow

To simulate the EPN/Agregation node topology you can execute the following.
Run `o2-tpc-miptrack-filter` workflow, it will start listening to `TPC/MIPS/0`, waiting for tracks data to process.

```
o2-dpl-raw-proxy --dataspec A:TPC/MIPS/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" | o2-tpc-calib-dedx
```

Now, in a new shell, start `o2-tpc-miptrack-filter`.

```
o2-tpc-track-reader --disable-mc | o2-tpc-miptrack-filter | o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec downstream:TPC/MIPS -b
```
