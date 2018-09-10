## CCDB API

The CCDB API class (`CcdbApi`) is implemented using libcurl and gives
access to the CCDB via its REST api.

Usage :
```
// init
CcdbApi api;
map<string, string> metadata; // can be empty
api.init("http://ccdb-test.cern.ch:8080");
// store
auto h1 = new TH1F("object1", "object1", 100, 0, 99);
api.store(h1, "Test/Detector", metadata);
// retrieve
auto h1back = api.retrieve("Test/Detector", metadata);

```

## Conditions MQ

Conditions MQ is a client/server CCDB implementation for O2. Currently the implementation supports two backends, an OCDB and a Riak one.

### OCDB backend

To run the MQ server-client example with the MQ server replying with CDB objects to the client requests, the steps below should be followed:

* Create a local O2CDB instance using the following macro in <installation directory>/bin/config/:

```bash
root -l fill_local_ocdb.C
```

This will create "DET/Calib/Histo" calibration objects for a hundred runs in the subdirectory O2CDB/ under the current directory.

* In one shell run the server example:

```bash
conditions-server --id parmq-server --mq-config <installation directory>/bin/config/conditions-server.json --first-input-name local://<installation directory>/bin/config/O2CDB --first-input-type OCDB
```

* In a separate shell run the client example:

```bash
conditions-client --id parmq-client --mq-config <installation directory>/bin/config/conditions-client.json --data-source OCDB --object-path <installation directory>/bin/config/O2CDB
```

* We can also query the running conditions-server using any user code as
  demonstrated in `standalone-client` which works for an O2CDB
  generated from the unit test `testWriteReadAny`

### Riak backend

To run the MQ server-client example with the MQ server executing PUT or GET commands to a Riak cluster through an MQ broker, the steps below should be followed:

* In one shell run the server example:

```bash
conditions-server --id parmq-server --mq-config <installation directory>/bin/config/conditions-server.json
```

* In a separate shell run the client example:

```bash
conditions-client --id parmq-client --mq-config <installation directory>/bin/config/conditions-client.json --data-source Riak
```

List of optional client arguments:

- `operation-type` (default = "GET"): "PUT", "GET". Sets the operation type.
- `object-path` (default = "./OCDB/"). Sets the directory that holds the condition objects.
