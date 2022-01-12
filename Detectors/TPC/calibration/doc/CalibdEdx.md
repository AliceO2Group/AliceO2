<!-- doxy
\page refTPCcalibrationCalibdEdx dEdx Calibration
/doxy -->

# dEdx Calibration

The workflow `o2-tpc-miptrack-filter` is supposed to run in the EPNs, and will select only the track inside the defined cuts, the selected track in the streamed with the name `TPC/MIPS/0`. The cut values can be changed with these CLI options:

```
--min-momentum (= 0.4)
--max-momentum (= 0.6)
--min-clusters (= 60)       Minimum number of cluster in a track.
```

The workflow `o2-tpc-calibratordedx` should run in an aggregation node. It will fill dEdx histograms using the data sended by the `o2-tpc-miptrack-filter`, and will compute corrections for the dE/dx values for every time slot.

```
--tf-per-slot
--max-delay
--min-entries           Minimum number of entries per GEM stack to perform a fit

--min-entries-sector    Bellow the number of entries per stack every sector will be integrated before the fit
--min-entries-1d        Minimum entries per stack to perform a 1D fit, bellow it only calculate the mean
--min-entries-2d        Mininum entries per stack to perform a 2D fit

--dedxbins              Number of dE/dx bins
--zbins                 Number of Z bins
--angularbins           Number of angular bins, for values like Tgl and Snp
--min-dedx              Min. dE/dx value
--max-dedx              Max. dE/dx value

--file-dump             Save calibration correction to a file
--field                 Magnetic field in kG, need for track propagations, this value will be overwritten if a grp file is present
```

The workflow `o2-tpc-calibdedx` is similar to `o2-tpc-calibratordedx`, but only compute correction for a single set of data.

## Executing

The full dE/dx calibration workflow can be executed in the following way:

```
o2-tpc-track-reader --disable-mc | o2-tpc-miptrack-filter | o2-tpc-calibrator-dedx --file-dump --min-entries 100 --tf-per-slot 10
```

Where we enabled the option to save the corrections to a file, set the min. entires per time slot to 100 and defined that the time slots should have 10 time frames.

## Simulating EPN workflow

To simulate the EPN/Agregation node topology you can execute the following.
Run the `o2-tpc-miptrack-filter` workflow, it will start listening to `TPC/MIPS/0`, waiting for tracks data to process.

```
o2-dpl-raw-proxy --dataspec A:TPC/MIPS/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" | o2-tpc-calib-dedx
```

Now, in a new shell, start the `o2-tpc-miptrack-filter`.

```
o2-tpc-track-reader --disable-mc | o2-tpc-miptrack-filter | o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec downstream:TPC/MIPS -b
```
