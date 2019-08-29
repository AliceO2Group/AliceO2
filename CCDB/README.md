\page refCCDB Module 'CCDB'

# CCDB

The Conditions and Calibration DataBase provides a REST API which can be used to list, store and retrieve objects.

The CCDB API class (`CcdbApi`) is implemented using libcurl and gives C++ API
access to the CCDB via its REST api. The API can be also used to create a snapshot
of conditions objects on the local disc and retrieve the objects therefrom. This can be useful
in circumstances of reduced or no network connectivity.

There are currently 3 different kinds of store/retrieve functions, which we expect to unify in the immediate future:
1. simple `store/retrieve` API serializing a `TObject` as a simple ROOT `TMessage`.
2. `storeAsTFile/retrieveFromTFile` API serializing a `TObject` in a ROOT `TFile` with the advantage 
   of keeping the data together with the ROOT streamer info in the same place.
3. A strongly-typed `storeAsTFileAny<T>/retrieveFromTFileAny<T>` API allowing to handle any type T 
   having a ROOT dictionary. We encourage to use this API by default.


## Central and local instances of the CCDB

There is a test central CCDB at [http://ccdb-test.cern.ch:8080](http://ccdb-test.cern.ch:8080). Feel free to use it. If you prefer to use a local instance, you can follow the instructions [here](https://docs.google.com/document/d/1_GM6yY7ejVEIRi1y8Ooc9ongrGgZyCiks6Ca0OAEav8).

## Access with a browser

If you access the CCDB with a web browser, add `/browse` at the end of the URL to have a user readable interface. Moreover, using `/browse/?report=true` will provide details on the number of files and the size of the folders (e.g. http://ccdb-test.cern.ch:8080/browse/?report=true).

## Example Usage

* storing / retrieving TObjects with TMessage blobs
```c++
// init
CcdbApi api;
map<string, string> metadata; // can be empty
api.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
// store
auto h1 = new TH1F("object1", "object1", 100, 0, 99);
api.store(h1, "Test/Detector", metadata);
// retrieve
auto h1back = api.retrieve("Test/Detector", metadata);
```

* storing / retrieving arbitrary (non TObject) classes

```c++
// init
CcdbApi api;
map<string, string> metadata; // can be empty
api.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
// store abitrary user object in strongly typed manner
auto deadpixels = new o2::FOO::DeadPixelMap();
api.storeAsTFileAny(deadpixels, "FOO/DeadPixels", metadata);
// read like this (you have to specify the type)
auto deadpixelsback = api.retrieveFromTFileAny<o2::FOO::DeadPixelMap>("FOO/DeadPixels", metadata);
```

* creating a local snapshot and fetching objects therefrom

```c++
// init
CcdbApi api;
map<string, string> metadata; // can be empty
api.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
// create a local snapshot of everthing in or below the FOO folder valid for timestamp 12345
api.snapshot("FOO", "/tmp/CCDBSnapshot/", 12345);

// read from snapshot by saying
CcdbApi snapshotapi;
snaptshotapi.init("file:///tmp/CCDBSnapshot");

// reading still works just like this (you have to specify the type)
auto deadpixelsback = snapshotapi.retrieveFromTFileAny<o2::FOO::DeadPixelMap>("FOO/DeadPixels", metadata);
```

## Future ideas :

- [ ] offer API without need to pass metadata object
- [ ] deprecate TMessage based API
- [ ] code reduction or delegation between various storeAsTFile APIs
- [ ] eventually just call the functions store/retrieve once TMessage is disabled


# BasicCCDBManager

A basic higher level class `BasicCCDBManager` is offered for convenient access to the CCDB from
user code. This class
* Encapsulates the timestamp.
* Offers a more convenient `get` function to retrieve objects.
* Is a singleton which is initialized once and can be used from any detector code.

The class was written for the use-case of transport MC simulation. Typical usage should be like

```c++
// setup manager once (at start of processing) 
auto& mgr = o2::ccdb::BasicCCDBManager::instance();
mgr.setURL("http://ourccdbserverver.cern.ch");
mgr.setTimestamp(timestamp_which_we_want_to_anchor_to);


// in some FOO detector code (detector initialization)
auto& mgr = o2::ccdb::BasicCCDBManager::instance();
// just give the correct path and you will be served the object
auto alignment = mgr.get<o2::FOO::GeomAlignment>("/FOO/Alignment");
```

## Future ideas / todo:

- [ ] offer improved error handling / exceptions
- [ ] do we need a store method?
