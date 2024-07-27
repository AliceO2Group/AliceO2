<!-- doxy
\page refMUONMIDCalibrationMacros MID Calibration macros
/doxy -->

# MID calibration macros

This repository contains macros that can be used to handle calibration objects in the CCDB.

## ccdbUtils.C

This macro allows to query a series of MID CCDB objects in the CCDB as well as produce default objects.
The basic usage is the following:

Query the list of bad channels from the official CCDB:

```shell
root -l
.x ccdbUtils.C("querybad",1721252719000,"mask.txt",true,"http://alice-ccdb.cern.ch")
```

Upload the default list of fake dead channels to the local CCDB (assuming that an instance of the local CCDB is running):

```shell
root -l
.x ccdbUtils.C("uploadfake",1,"mask.txt",true,"localhost:8080")
```

The macro is also used to keep track of the fake dead channels, which are generated in `makeFakeDeadChannels()`.

## buils_rejectList.C

This macro analyses the Quality flag and the Occupancy plot in the QCCDB and searches for issues appearing in the middle of the run, e.g. local board(s) that become noisy and are then excluded from the data taking by the user logic of the CRU.
It then correlate this information with the GRPECS object in the CCDB in order to create a reject list that will allow to mask the faulty local board(s) from slightly before the problem appeared till the end of the run.

Notice that the QCDB is not directly reachable from outside CERN. In that case one needs to first create an ssh tunnel:

```shell
ssh -L 8083:ali-qcdb-gpn.cern.ch:8083 lxtunnel.cern.ch
```

We also advice to have an instance of the local CCDB running, so that the reject list objects will be saved there.
One can then scan for issues with:

```shell
root -l
build_rejectlist.C+(1716436103391,1721272208000,"localhost:8083")
```

Where the first number is the start timestamp for the scan, and the second is the end timestamp of the scan.
For each problem found, the macro will ask if one wants to upload the reject list to the (local) CCDB or not.
