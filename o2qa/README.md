O2qa
=======

Quality assurance software prototype.

### Prerequisites
0. Installed AliceO2 software.
1. Set the environment variable SIMPATH to your FairSoft installation directory.
2. Set the environment variable FAIRROOTPATH to your FairRoot installation directory.

It is a good practice to run config.sh script from AliceO2 build directory to set all others variables such as PATH etc.

### Overwiev
This is a quality assurance software prototype for AliceO2 project. It uses FairMQ framework to provide distributed environment.

Project consists of three modules:
## Producer - produces histograms or trees
Run example for histogram:
```bash
runProducer -histogram exampleHistogramPrefixName exampleHistogramTitle -10 10
```
The last two parameters describes minimal and maximal values of x axis.

Run example for tree:
```bash
runProducer -tree treeName_ treeTitle_ 4 1000
```
The fourth parameter gives number of branches created in the tree.
The last parameter is the number of entries in each branch. 

## Merger - merges received objects by titles. 
It can merge both trees and histograms with the same title.

Run example:
```bash
runMergerDevice
```
## Viewer - provides visualization of merged objects
Run example:
```bash
runViewerDevice
```
Viewer can received additional parameter which describes drawing option given to Draw function of TObject class.

### Compile software
1. Go to build folder of AliceO2 software
2. cmake ../
3. cd o2qa
4. make

### Unit tests
All modules are provided with unit tests written in BOOST test framework. Each module has it's tests in "Tests" directory.
To run all unit tests type ```ctest ```
