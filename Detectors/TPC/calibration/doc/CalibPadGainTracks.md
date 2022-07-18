<!-- doxy
\page refTPCcalibrationCalibPadGainTracks Residual gainmap calibration
/doxy -->

# Residual gainmap calibration
The workflow `o2-tpc-calib-gainmap-tracks` will run on the EPNs. In this workflow the cluster charge qMax is normalized to the dE/dx of the track and stored for each pad in a histogram. As input tracks and clusters are required which can be provided by the `o2-tpc-file-reader` workflow. The options for the pad-by-pad histograms and the cuts for the tracks can be set with these CLI options:


```
Histogram options:
--nBins             Number of bins per histogram
--reldEdxMin        Minimum x coordinate of the histogram for Q/(dE/dx)
--reldEdxMax        Maximum x coordinate of the histogram for Q/(dE/dx)
--underflowBin      Using under flow bin
--overflowBin       Using under flow bin
--useEveryNthTF     Use only every nth TF

Track cuts:
--momMin            Minimum momentum of the tracks
--momMax            Maximum momentum of the tracks
--etaMax            Maximum eta of the tracks
--minClusters       Minimum number of clusters of tracks

Other:
--field             Magnetic field in kG, need for track propagations, this value will be overwritten if a grp file is present
--debug             Writing debug files when objects are send
--publish-after-tfs Number of TFs after which the pad-by-pad histograms are send
```

The workflow `o2-tpc-calibrator-gainmap-tracks` should run in an aggregation node. It will use the pad-by-pad histograms sent by `o2-tpc-calib-gainmap-tracks` to create the pad-by-pad residual gainmap for every time slot. The following options are available:

```
--tf-per-slot   Number of TFs per calibration slot
--max-delay     Number of slots in past to consider

--min-entries   Minimum number of entries per pad-by-pad histogram which are required
--lowTrunc      Lower truncation range for calculating the residual gain from the pad-by-pad histogram
--upTrunc       Upper truncation range for calculating the residual gain from the pad-by-pad histogram

--file-dump     Save calibration class object to a file
```

## Executing

The full residual gainmap extraction workflow can be executed in the following way:

```
o2-tpc-file-reader --input-type "clusters,tracks" --disable-mc \
| o2-tpc-calib-gainmap-tracks --publish-after-tfs 100 --overflowBin true --condition-tf-per-query -1 --debug true \
| o2-tpc-calibrator-gainmap-tracks --min-entries 0 --tf-per-slot 100 --file-dump true --shm-segment-size 100000000000 -b
```


## Simulating EPN workflow

To simulate the EPN/Agregation node topology one can execute the following:

For the Aggregator:

```
o2-dpl-raw-proxy --dataspec A:TPC/TRACKGAINHISTOS/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq" \
| o2-tpc-calibrator-gainmap-tracks --ccdb-uri "http://localhost:8080" --min-entries 0 --tf-per-slot 100
```

For the EPN start the `o2-tpc-calib-gainmap-tracks` workflow in a new shell:

```
o2-tpc-file-reader --input-type "clusters,tracks" --disable-mc \
| o2-tpc-calib-gainmap-tracks --publish-after-tfs 100 --overflowBin true --condition-tf-per-query -1 \
| o2-dpl-output-proxy --channel-config "name=downstream,method=connect,address=tcp://localhost:30453,type=push,transport=zeromq" --dataspec downstream:TPC/TRACKGAINHISTOS -b
```
