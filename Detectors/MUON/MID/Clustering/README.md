# Running MID clustering

The MID clustering takes as input the "digits" and returns the resulting clusters.
There is no charge information in MID, so a uniform charge distribution is assumed.

The "digits" data are not yet implemented for MID.
However, MID is an upgrade of the existing MTR detector, with unchanged segmentation.
This means that we can test the device right now using the MTR digits as input (that can be broadcast with the HLT wrapper).
The details of the decoding of the digits will change once we will be able to generate the digits in the new format.
This will be nevertheless a minor modification of the code, restricted to few lines of the ClusterizerDevice, and will be done in the future.

For the moment it is therefore possible to run the digitizer on the MTR digits in the way explained below.

## Getting the digits
To produce a digit file suitable as input of the HLT wrapper, the utility O2muon was created in the [alo project](https://github.com/mrrtf/alo), see in particular [r23](https://github.com/mrrtf/alo/tree/master/r23).
The utility allows to generate a digit file from raw data.

## Execution
### Start digits reader
See instructions [here](https://github.com/mrrtf/alo/tree/master/dhlt).

### Start clusterizer
In another terminal:
```bash
runMIDclusterizer --id 'MIDclusterizer' --mq-config "$O2_ROOT/etc/config/runMIDclusterizer.json"
```
