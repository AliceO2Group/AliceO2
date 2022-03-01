<!-- doxy
\page refDetectorsTRDcalibration TRD calibration
/doxy -->

# TRD calibration in O2

## vDrift and ExB calibration

The drift velocity and ExB angle are calibrated using input from the global tracking, namely ITS-TPC-TRD and TPC-TRD tracks. The latter is currently disabled, but can be enabled to have more statistics easily. From the global tracks we obtain the association of TRD tracklets to tracks. In the calibration procedure  a refit is performed to get TRD only tracks and the angular deviations between those TRD tracks and the associated tracklets are stored in histograms (see class TrackBasedCalib). This is done on the EPNs during the reconstruction. The histograms are send to aggregator nodes which calculate for each TRD chamber the drift velocity and ExB angle based on those histograms (see class CalibratorVdExB).

To run the calibration one can start the following workflow:
    o2-trd-global-tracking -b --disable-root-output | o2-calibration-trd-vdrift-exb -b --calib-vdexb-calibration '--tf-per-slot 5 --min-entries 50000 --max-delay 90000'
You can get information on the meaning of the parameters by running `o2-calibration-trd-vdrift-exb -b --help full`

If you want to run the calibration from a local file with residuals, trdangreshistos.root, you can run:
    o2-calibration-trd-vdrift-exb -b --enable-root-input --calib-vdexb-calibration '--tf-per-slot 1 --min-entries 50000'

## DCS data points
