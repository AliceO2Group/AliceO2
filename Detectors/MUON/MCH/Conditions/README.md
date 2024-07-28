<!-- doxy
\page refDetectorsMUONMCHConditions Conditions
/doxy -->

# MCH Conditions

The condition data we have are :

- the DCS datapoints (for HV and LV)
- the Bad Channel list (obtained from pedestal calibration runs)
- the Reject list (manual input)

Those objects are stored at the following CCDB paths :

- MCH/Calib/HV
- MCH/Calib/LV
- MCH/Calib/BadChannel
- MCH/Calib/RejectList

The BadChannel and RejectList objects can be uploaded, e.g. for debug purposes, using the `o2-mch-bad-channels-ccdb` program :

```shell
$ o2-mch-bad-channels-ccdb --help
This program get/set MCH bad channels CCDB object
Usage:
  -h [ --help ]                         produce help message
  -c [ --ccdb ] arg (=http://localhost:6464)
                                        ccdb url
  --starttimestamp arg (=1677687518645) timestamp for query or put -
                                        (default=now)
  --endtimestamp arg (=1677773918645)   end of validity (for put) - default=1
                                        day from now
  -l [ --list ]                         list timestamps, within the given
                                        range, when the bad channels change
  -p [ --put ]                          upload bad channel object
  -r [ --referenceccdb ] arg (=http://alice-ccdb.cern.ch)
                                        reference ccdb url
  -u [ --upload-default-values ]        upload default values
  -t [ --type ] arg (=BadChannel)       type of bad channel (BadChannel or
                                        RejectList)
  -q [ --query ]                        dump bad channel object from CCDB
  -v [ --verbose ]                      verbose output
  -s [ --solar ] arg                    solar ids to reject
  -d [ --ds ] arg                       dual sampas indices to reject
  -e [ --de ] arg                       DE ids to reject
  -a [ --alias ] arg                    DCS alias (HV or LV) to reject
```

For instance, to create in a local CCDB a RejectList object which declares solar number 32 as bad, from Tuesday 1 November 2022 00:00:01 UTC to Saturday 31 December 2022 23:59:59, use :

```shell
$ o2-mch-bad-channels-ccdb -p -s 32 -t RejectList --starttimestamp 1667260801000 --endtimestamp 1672531199000
```

The program will search the reference CCDB (defined with `--referenceccdb`) for existing objects valid during this period and propose you to either overwrite them or update them. In the first case, a single object will be created, valid for the whole period, containing only the new bad channels. In the second case, as many objects as necessary will be created with appropriate validity ranges, adding the new bad channels to the existing ones.
