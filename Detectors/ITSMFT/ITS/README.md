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

