<!-- doxy
\page refITS ITS
/doxy -->

# ITS

## Workflow and executables

*   `o2-its-digi2raw`: creation of raw data from MC. Requires digitized ITS data (as well as an access to the GRP data). Allows creation of raw data file per layer (default) or per CRU. The configuration file for the input to the `o2-raw-file-reader-workflow` will be automatically created in the output directory.

*   `o2-its-reco-workflow`: reconstruction of ITS tracks starting from simulated digits.

*   `o2-itsmft-stf-decoder-workflow`: raw data STF decoder and clusterizer. Provides either cluster or digits or both. Supports multi-threading.

Can be extended to reconstruction from the raw data by disabling the digits reader and piping it to the output of the STF reader:

```bash
# To decode digits from the raw (simulated) STF and send feed to to the workflow for further clusterization and reconstruction:
o2-raw-file-reader-workflow --loop 5 --delay 3 --conf ITSraw.cfg | o2-itsmft-stf-decoder-workflow --digits --no-clusters | o2-its-reco-workflow --disable-mc --digits-from-upstream
```

```bash
# To decode/clusterize the STF and feed directly the clusters into the workflow:
o2-raw-file-reader-workflow --loop 5 --delay 3 --conf ITSraw.cfg | o2-itsmft-stf-decoder-workflow | o2-its-reco-workflow --disable-mc --clusters-from-upstream
```

```bash
#If needed, one can request both digits and clusters from the STF decoder:
o2-raw-file-reader-workflow --loop 5 --delay 3 --conf ITSraw.cfg | o2-itsmft-stf-decoder-workflow --digits  | o2-its-reco-workflow --disable-mc --digits-from-upstream --clusters-from-upstream
```

## Tracking

```bash
# main tracking workflow, by default run CookedSeed Tracker, with --trackerCA uses CA-tracker
o2-its-reco-workflow
```

For the synchronous mode one can apply selection on the ROF multiplicity either in therms of signal clusters and/or number of contributing tracklets in seeding vertices
`FastMultEst` implements the fast estimator of signal clusters multiplicity either via free noise + signal fit or via signal fit with noise level imposed (via `FastMultEstConfig.imposedNoisePerChip`).
Multiplicity is provided as Ncl. per layer modulo acceptance correction from the FastMultEstConfig.
The latter provides the settings for the estimator as well as the low/high cuts on the estimate mult. and (if these cuts are passed) eventual cuts on the vertices multiplicity used to seed the ITS tracks. 
For example, the command:
```cpp
o2-its-reco-workflow --configKeyValues "fastMultConfig.cutMultClusLow=50;fastMultConfig.cutMultClusHigh=4000;fastMultConfig.cutMultVtxHigh=1000"
```
will track only ROFs with N signal clusters between 50 and 4000 and will consider seeding vertices with multiplicity below 1000 tracklets.

<!-- doxy
* \subpage refITScalibration
/doxy -->

