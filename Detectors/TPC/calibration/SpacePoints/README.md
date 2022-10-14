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
