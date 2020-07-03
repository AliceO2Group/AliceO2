<!-- doxy
\page refDetectorsMUONMCHWorkflow Workflows
/doxy -->

# MCH Workflows

## Raw data decoding:

`o2-mch-raw-to-digits-workflow`

## Pre-clustering:

`o2-mch-digits-to-preclusters-workflow`

## Pre-clustering sink:

`o2-mch-preclusters-sink-workflow`

## Example of DPL chain:

`o2-raw-file-reader-workflow --conf file-reader.cfg --loop 0  -b | o2-mch-raw-to-digits-workflow -b | o2-mch-digits-to-preclusters-workflow -b | o2-mch-preclusters-sink-workflow -b`

where the `file-reader.cfg` looks like this:

    [input-0]
    dataOrigin = MCH
    dataDescription = RAWDATA     
    filePath = /home/data/data-de819-ped-raw.raw
