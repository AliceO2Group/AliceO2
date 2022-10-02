<!-- doxy
\page refDetectorsTRDcalibration TRD calibration
/doxy -->

# TRD calibration in O2

## vDrift and ExB calibration

The drift velocity and ExB angle are calibrated using input from the global tracking, namely ITS-TPC-TRD and TPC-TRD tracks.
The latter is currently disabled, but can be enabled to have more statistics easily.
From the global tracks we obtain the association of TRD tracklets to tracks.
In the calibration procedure a refit is performed to get TRD only tracks and the angular deviations between those TRD tracks and the associated tracklets are stored in histograms (see class TrackBasedCalib).
This is done on the EPNs during the reconstruction.
The histograms are send to aggregator nodes which calculate for each TRD chamber the drift velocity and ExB angle based on those histograms (see class CalibratorVdExB).

To run the calibration one can start the following workflow:

    o2-trd-global-tracking -b --disable-root-output --enable-trackbased-calib | o2-calibration-trd-vdrift-exb -b --calib-vdexb-calibration '--tf-per-slot 5 --min-entries 50000 --max-delay 90000'

*Hint: You can get information on the meaning of the parameters by running `o2-calibration-trd-vdrift-exb -b --help full`*

If you want to run the calibration from a local file with residuals, trdangreshistos.root, you can run:

    o2-calibration-trd-vdrift-exb -b --enable-root-input --calib-vdexb-calibration '--tf-per-slot 1 --min-entries 50000'

Additionally it is possible to perform the calibrations fit manually per chamber if you have TPC-TRD or ITS-TPC-TRD tracks, you can run:

    o2-trd-global-tracking -b --enable-trackbased-calib

This produces `trdangreshistos.root` which holds the residuals of the angles and differences.
Then run the macro `Detectors/TRD/calibration/macros/manualCalibFit.C`.
This produces a file of similar name with the fitted data and prints out the fit results.
This is equivalent to running:

    o2-calibration-trd-vdrift-exb -b --enable-root-input '--tf-per-slot 1 --min-entries 2000 --enable-root-output'

You can plot the calibration values for VDrift and ExB for a given Run by using the macro at `Detectors/TRD/cailbration/macros/plotVdriftExB.C`.
This produces a root file of similar name which holds the time-series plots (reference the macro).

It may also be worthwhile to look at the correlation of the deflection and slope of tracks and tracklets since this is good indicator how well the calibration works.
For that you can use the macro at `Detectors/TRD/calibration/macros/makeDeflectionCorrelation.C` and to you have to include the '--enable-qc' flag.

    o2-trd-global-tracking -b --enable-qc

In the macro you have to point to the right Run and potentially the correct CCDB.

## DCS data points

To process the DCS data points for the TRD the list of aliases for example "trd_gaschromatographXe" has to be available in the CCDB.
This can be achieved with the macro `Detectors/TRD/calibration/macros/makeTRDCCDBEntryForDCS.C`.
The full list of aliases is available in jira (https://alice.its.cern.ch/jira/browse/TRD-109).
With the list of aliases defined one can run the `o2-calibration-trd-dcs-sim-workflow` which provides DCS DPs for all possible aliases and sends them on via DPL.
Attaching the `o2-calibration-trd-dcs-workflow` will include the processing of these data points.
For testing purposes this is sufficient. In case also the CCDB should be populated the `o2-calibration-ccdb-populator-workflow` has to be appended.
Via the `--ccdb-path` flag this can also be configured to write to a local CCDB for testing.

So, in order to test the workflow independent of the actual CCDB using a local instance one can do:

    root $O2_ROOT/share/macro/makeTRDCCDBEntryForDCS.C+
    o2-calibration-trd-dcs-sim-workflow -b --delta-fraction 0.5 --max-timeframes 10 | o2-calibration-trd-dcs-workflow -b --ccdb-path http://localhost:8080 --use-ccdb-to-configure --processor-verbosity 1 | o2-calibration-ccdb-populator-workflow -b --ccdb-path http://localhost:8080

The macro `readTRDDCSentries.C` shows an example how to read one of the created CCDB objects after processing.
