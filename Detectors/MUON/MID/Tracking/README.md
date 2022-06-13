<!-- doxy
\page refMUONMIDTracking MID Tracking
/doxy -->

# Running MID Tracking

The MID tracking takes as input the clusters and returns a vector of reconstructed tracks.
The magnetic field in the trigger chambers, placed behind an iron wall, is negligible, so a straight track is used.

The algorithm has some configurable parameters in [TrackerParam.h](include/MIDTracking/TrackerParam.h).
Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change them from the command line.

## Execution
### Start clusterizer
See instructions [here](../Clustering/README.md).

### Start tracker
In another terminal:
```bash
runMIDTracking --id 'MIDtracker' --mq-config "$O2_ROOT/etc/config/runMIDtracking.json"
```
