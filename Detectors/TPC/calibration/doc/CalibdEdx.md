<!-- doxy
\page refTPCcalibrationCalibdEdx dEdx Calibration
/doxy -->

# dEdx Calibration

For now we only have a test implementation of this.

The workflow `o2-tpc-miptrack-filter` is supposed to run in the EPNs, and will select only the track inside the defined cuts, the selected track in the streamed with the name `TPC/MIPS/0`. The cut values can be changed with these CLI options:

```
--min-momentum (= 0.4)
--max-momentum (= 0.6)
--min-clusters (= 60)       Minimum number of cluster in a track.
```

The workflow `o2-tpc-calibdedx` should run in an aggregation node. It will fill dEdx histograms using the data sended by the `o2-tpc-miptrack-filter`, and will find the MIP for every one of them.

## Simulating

Run the `o2-tpc-miptrack-filter` workflow, it will start listening to `TPC/MIPS/0`, waiting for tracks data to process.

```
o2-dpl-raw-proxy --dataspec A:TPC/MIPS/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" | o2-tpc-calib-dedx
```

Now, in a new shell, start the `o2-tpc-miptrack-filter`.

```
o2-tpc-track-reader --disable-mc | o2-tpc-miptrack-filter | o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec downstream:TPC/MIPS -b
```
