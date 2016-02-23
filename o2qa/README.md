O2qa
=======

Quality assurance software prototype.

### Prerequisites
0. Installed AliceO2 software.
1. Set the environment variable SIMPATH to your FairSoft installation directory.
2. Set the environment variable FAIRROOTPATH to your FairRoot installation directory.

It is a good practice to run config.sh script from AliceO2 build directory to set all others variables such as PATH etc.

### Overwiev
This is a quality assurance software prototype for AliceO2 software. It uses FairMQ framework to provide distributed environment.

Project consists of three modules:
## Producer - produces histograms and trees
Run example:
```bash
runProducer -10 10 exampleHistogramPrefixName exampleHistogramTitle
```
First two arguments provides information about x axis range.
## Merger - merges received objects by titles
Run example:
```bash
runMergerDevice
```
## Viewer - provides visualization of merged objects
Run example:
```bash
runViewerDevice
```
### Compile software
1. Go to build folder of AliceO2 software
2. cmake ../
3. cd o2qa
4. make

### Unit tests
All modules are provided with unit tests written in BOOST test framework. Each module has it's tests in "Tests" directory.
To run all unit tests type ```ctest ```