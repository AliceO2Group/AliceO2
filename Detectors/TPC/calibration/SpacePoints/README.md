# TPC average distortion correction

The average correction for TPC space charge distortions is based on residuals of TPC clusters with respect to global ITS-TPC-TRD-TOF tracks.
Multiple DPL devices at different stages are involved as can be seen in the sketch below:

![TPC-SCD-scheme](https://user-images.githubusercontent.com/26281793/195542425-136852a3-f9e9-4c85-9490-f2cfb262686c.png)

On each EPN the residuals of the TPC clusters wrt global tracks are collected and sent for each TF to the aggregator. Here the residuals are put into voxels and for each voxel some statistics are collected (number of entries and center of gravity in XYZ for all residuals). The aggregator produces one file per calibration slot and this file will eventually shipped to EOS.

The workflow on the EPNs can be configured to sent in addition to the residuals also the track data by settings
```
CALIB_TPC_SCDCALIB_SENDTRKDATA=1
```
Currently this is the default and can be disabled by putting `CALIB_TPC_SCDCALIB_SENDTRKDATA=0` as extra ENV variable.

The workflow running on the aggregator is at the moment causing backpressure on the calibration nodes. In order to mitigate this the ROOT compression can be disabled which should make the processing much faster. This is added as option to the `o2-calibration-residual-aggregator` workflow.

```
ARGS_EXTRA_PROCESS_o2_calibration_residual_aggregator="--compression 0"
```
Note, that currently writing to EOS is disabled. Once the workflow is validated it should be enabled in `aggregator-workflow.sh`: https://github.com/AliceO2Group/AliceO2/blob/cb98b7e5f036024dfce73abf2b6bdb3ad432d1b4/prodtests/full-system-test/aggregator-workflow.sh#L177-L181


## Running the calibration offline from ROOT file input

For running the track interpolation offline which creates the unbinned residuals one needs to have the following inputs in the local directory as ROOT files:

- clusters, tracks, track matches for all barrell detectors ITS, TPC, TRD and TOF
- the TF ID information typically stored in o2_tfidinfo.root

Now, one can rerun the interpolation workflow piping its output directly into the residual aggregator which creates the o2tpc_residuals*.root files with the specified output (residuals binned/unbinned, track information):

```
o2-tpc-scdcalib-interpolation-workflow -b --send-track-data --disable-mc --hbfutils-config o2_tfidinfo.root | \
o2-calibration-residual-aggregator -b --enable-track-input  --output-type unbinnedResid,binnedResid,trackParams --output-dir $PWD --disable-root-input
```

Since the workflow loads also the TPC native clusters which are rather large, you will most likely need to increase the SHM size via adding `--shm-segment-size 32000000000` to both workflows.

#### Track selection

Track selections are applied at the interpolation stage. This can be steered via configurable parameters. For example add

```
--configKeyValues "scdcalib.maxTracksPerCalibSlot=-1;scdcalib.minTPCNCls=70"
```
in order to process all global tracks which are available per TF and have at least 70 TPC clusters.
For all options please have a look at `Detectors/TPC/calibration/SpacePoints/include/SpacePoints/SpacePointsCalibConfParam.h`

### Creating the distortion map from unbinned residuals

When the unbinned residuals are available it is possible to create the distortion map directly from those and on-the-fly change the track selection, e.g. taking into account only ITS-TPC-TRD tracks or only tracks with a very high quality. Use the `Detectors/TPC/calibration/SpacePoints/macro/staticMapCreator.C` macro in compiled mode. You will need to specify the input file (single ROOT file or list of files) and the run number. Optionally you can specify the output file name or the track type to be used (default is ALL which are available as input, use e.g. "ITS-TPC-TRD" for only this type).

The track selection is steered via configurable param. Before running the macro it's best to create a configuration file which can then be adapted to ones needs:
```
const o2::tpc::SpacePointsCalibConfParam& params = o2::tpc::SpacePointsCalibConfParam::Instance();
params.writeINI("scdconfig.ini", "scdcalib"); // to write default parameters to a file
```
For example to request only tracks with 5 ITS clusters, 2 TRD tracklets and a DCA < 10 cm set
```
minITSNCls=5
minTRDNTrklts=2
writeBinnedResiduals=true
useTrackData=true
cutOnDCA=true
maxDCA=10
```
It is also possible to select only TFs from a specific time range. The time range needs to be provided in ms. The epoch time in seconds for a given date can be determined e.g. via `date -d "Nov 22 2022 16:00"  +%s`. Don't forget to multiply by 1000 to get ms.
```
timeFilter=true
startTimeMS=1666669765254
endTimeMS=1666670365241
```
