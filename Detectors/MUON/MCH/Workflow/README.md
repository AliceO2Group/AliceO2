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

`o2-raw-file-reader-workflow --conf file-reader.cfg --loop 0 --message-per-tf -b | o2-mch-raw-to-digits-workflow -b | o2-mch-digits-to-preclusters-workflow -b | o2-mch-preclusters-sink-workflow -b`

where the `file-reader.cfg` looks like this:

    [input-0]
    dataOrigin = MCH
    dataDescription = RAWDATA     
    filePath = /home/data/data-de819-ped-raw.raw

or

`o2-mch-cru-page-reader-workflow --infile /home/data/data-de819-ped-raw.raw --nframes -1 --full-hbf -b | o2-mch-cru-page-to-digits-workflow -b | o2-mch-digits-to-preclusters-workflow -b | o2-mch-preclusters-sink-workflow -b`

The `o2-mch-cru-page-reader-workflow` is more tolerant to badly formatted RDHs or inconsistencies in the HBframe sequences, hance is more suitable for processing test data.
