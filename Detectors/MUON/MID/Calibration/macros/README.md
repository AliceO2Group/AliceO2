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

Upload the default list of fake dead channels to the local CCDB (assuming that an instance of the local CCDB is running, see below):

```shell
root -l
.x ccdbUtils.C("uploadfake",1,"mask.txt",true,"localhost:8080")
```

The macro is also used to keep track of the fake dead channels, which are generated in `makeFakeDeadChannels()`.

## build_rejectList.C

This macro analyses the quality flag and the occupancy plots in the QCCDB and searches for issues appearing in the middle of the run, e.g. local board(s) that become noisy and are then excluded from the data taking by the user logic of the CRU.
It then correlates this information with the GRPECS object in the CCDB in order to create a reject list that will allow to mask the faulty local board(s) from slightly before the problem appeared till the end of the run.

If a problem is found, the macro will ask to upload the reject list in the local CCDB.
For this, one needs to have the local CCDB up and running (see below).

The scan can be then performed with:

```shell
root -l
.x build_rejectlist.C+(1716436103391,1721272208000)
```

Where the first number is the start timestamp for the scan, and the second is the end timestamp of the scan.

### Running outside CERN network

Notice that the QCDB is not directly reachable from outside CERN network.
In that case one needs to first create an ssh tunnel:

```shell
ssh -L 8083:ali-qcdb-gpn.cern.ch:8083 lxtunnel.cern.ch
```

and then tell the macro to reach the QCDB via the tunneled local port:

```shell
root -l
.x build_rejectlist.C+(1716436103391,1721272208000,"localhost:8083")
```

### Add custom bad channels

The macro `build_rejectlist.C` scans the QCDB and the CCDB in search of issues.
However, the QCDB flag is based on local boards with empty signals.
It can happen that a local board is problematic, but not completely dead and, therefore, it is not correctly spotted by the macro.
It is therefore important to have a way to add the issues by hand.
This can be done with a json file in the form:

```json
{
    "startRun": 557251,
    "endRun": 557926,
    "startTT": 1726300235000,
    "endTT": 1726324000000,
    "rejectList": [
        {
            "deId": 4,
            "columnId": 2,
            "patterns": [
                "0x0",
                "0xFFFF",
                "0x0",
                "0x0",
                "0x0"
            ]
        },
        {
            "deId": 13,
            "columnId": 2,
            "patterns": [
                "0x0",
                "0xFFFF",
                "0x0",
                "0x0",
                "0x0"
            ]
        }
    ]
}
```

Where `startTT` and `endTT` are the timestamps in which the manual reject list will be built. To use the timestamps of start/end of the specified runs set `startTT` and `endTT` to 0 (or do not include them in the json).

The path to the file is then given to the macro with:

```shell
.x build_rejectlist.C+(1726299038000,1727386238000,"http://localhost:8083","http://alice-ccdb.cern.ch","http://localhost:8080","rejectlist.json")
```

The macro will then merge the manual reject list from the file with the reject list that it finds by scanning the QCDB and CCDB.

## Running the local CCDB

The local CCDB server can be easily built through alibuild.
As usual, one needs to be in the directory containing alidist and then run:

```shell
aliBuild build localccdb
```

The CCDB server can be then run with:

```shell
alienv enter localccdb/latest
startccdb -r "path_to_local_ccdb"
```

where `path_to_local_ccdb` is a directory in your local pc where the CCDB objects will be located.
