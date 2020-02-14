<!-- doxy
\page refCCDB Module 'CCDB'
/doxy -->

# CCDB

The Conditions and Calibration DataBase provides a REST API which can be used to list, store and retrieve objects.

The CCDB API class (`CcdbApi`) is implemented using libcurl and gives C++ API
access to the CCDB via its REST api. The API can be also used to create a snapshot
of conditions objects on the local disc and retrieve the objects therefrom. This can be useful
in circumstances of reduced or no network connectivity.

There are currently 2 different kinds of store/retrieve functions, which we expect to unify in the immediate future:
2. `storeAsTFile/retrieveFromTFile` API serializing a `TObject` in a ROOT `TFile`.
3. A strongly-typed `storeAsTFileAny<T>/retrieveFromTFileAny<T>` API allowing to handle any type T 
   having a ROOT dictionary. We encourage to use this API by default.

## Central and local instances of the CCDB

There is a test central CCDB at [http://ccdb-test.cern.ch:8080](http://ccdb-test.cern.ch:8080). Feel free to use it. If you prefer to use a local instance, you can follow the instructions [here](https://docs.google.com/document/d/1_GM6yY7ejVEIRi1y8Ooc9ongrGgZyCiks6Ca0OAEav8).

## Access with a browser

If you access the CCDB with a web browser, add `/browse` at the end of the URL to have a user readable interface. Moreover, using `/browse/?report=true` will provide details on the number of files and the size of the folders (e.g. http://ccdb-test.cern.ch:8080/browse/?report=true).

## Example Usage

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
// read like this to get the headers as well, and thus the metadata attached to the object 
map<string, string> headers;
auto deadpixelsback = api.retrieveFromTFileAny<o2::FOO::DeadPixelMap>("FOO/DeadPixels", metadata /* constraint the objects retrieved to those matching the metadata */, -1 /* timestamp */, &headers /* the headers attached to the returned object */); 
// finally, use this method to retrieve only the headers (and thus the metadata)
std::map<std::string, std::string> headers = f.api.retrieveHeaders("FOO/DeadPixels", f.metadata); 
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
- [ ] offer API without need to pass metadata object
- [ ] code reduction or delegation between various storeAsTFile APIs
- [ ] eventually just call the functions store/retrieve once TMessage is disabled

## Note on the use of TFile to store data

The reason for using a TFile to store the data in the CCDB is because it has the advantage of keeping the data together with the ROOT streamer info in the same place and because it makes it easy to open objects directly from a ROOT session. The streamers enable class schema evolution.

## Command line tools

A few prototypic command line tools are offered. These can be used in scriptable generic workflows
and facilitate the following tasks:

  1. Upload and annotate a generic C++ object serialized in a ROOT file
  
     ```bash
     o2-ccdb-upload -f myRootFile.root --key histogram1 --path /Detector1/QA/ --meta "Description=Foo;Author=Person1;Uploader=Person2"
     ```
     This will upload the object serialized in `myRootFile.root` under the key `histogram1`. Object will be put to the CCDB path `/Detector1/QA`.
     For full list of options see `o2-ccdb-upload --help`.
  
  2. Download a CCDB object to a local ROOT file (including its meta information)
  
     ```bash
     o2-ccdb-downloadccdbfile --path /Detector1/QA/ --dest /tmp/CCDB --timestamp xxx
     ```
     This will download the CCDB object under path given by `--path` to a directory given by `--dest` on the disc.
     (The final filename will be `/tmp/CCDB/Detector1/QA/snapshot.root` for the moment).
     All meta-information as well as the information associated to this query will be appended to the file.
     
     For full list of options see `o2-ccdb-downloadccdbfile --help`.
  
  3. Inspect the content of a ROOT file and print summary about type of contained (CCDB) objects and its meta information
  
     ```bash
     o2-ccdb-inspectccdbfile filename
     ```
     Lists all keys and stored types in a ROOT file (downloaded with tool `o2-ccdb-downloadccdbfile`) and prints a summary about the attached meta-information.


### TODO command line tools

- [ ] combine all tools into a single swiss-knife executable?
- [ ] offer more tools mapping all REST functionality of the server: be able to browse, list, query online content, etc.
- [ ] provide error diagnostics; different level of verbosity
- [ ] provide json interaction modes (give meta-information; retrieve meta-information)
- [ ] use meta-info filters when downloading
