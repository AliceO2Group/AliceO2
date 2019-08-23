\page refCCDB Module 'CCDB'

# CCDB

The Conditions and Calibration DataBase provides a REST API which can be used to list, store and retrieve objects. 

The CCDB API class (`CcdbApi`) is implemented using libcurl and gives
access to the CCDB via its REST api.

## Central and local instances of the CCDB

There is a test central CCDB at [http://ccdb-test.cern.ch:8080](http://ccdb-test.cern.ch:8080). Feel free to use it. If you prefer to use a local instance, you can follow the instructions [here](https://docs.google.com/document/d/1_GM6yY7ejVEIRi1y8Ooc9ongrGgZyCiks6Ca0OAEav8).

## Access with a browser

If you access the CCDB with a web browser, add `/browse` at the end of the URL to have a user readable interface. Moreover, using `/browse/?report=true` will provide details on the number of files and the size of the folders (e.g. http://ccdb-test.cern.ch:8080/browse/?report=true).

## Example Usage 
```
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
