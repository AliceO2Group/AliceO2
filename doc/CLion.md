Here are a number of instructions and tips on how to use at best the IDE called [CLion from Jetbrains](https://www.jetbrains.com/clion/).

## Installation

Download from the [website](https://www.jetbrains.com/clion/).

## License

We are currently negotiating new free licenses for the core members of the collaboration. See [here](https://alice.its.cern.ch/jira/browse/ATO-386).

## AliBuild integration

The difficulty here lies with the fact that the O2 software needs to be configures with cmake using a number of variables that point to the installation of its dependencies.
These instructions have to be carried out only once. 

1. Build O2 using alibuild as usual
2. Open the log file that should be in ~/alice/sw/BUILD/O2-latest/log
3. Near the top of the file identify the line starting with `+ cmake`. 
4. Copy the arguments starting from `-DFairRoot` till the end of the command. 
5. Paste it in an editor 
6. The annoying part starts : replace all references to a version by "latest", e.g. : 
   * `-DFairRoot_DIR=/home/local/alice/sw/slc7_x86-64/FairRoot/alice-dev-20170711-1` -> `-DFairRoot_DIR=/home/local/alice/sw/slc7_x86-64/FairRoot/latest`
   * `-DPROTOBUF_PROTOC_EXECUTABLE=/home/local/alice/sw/slc7_x86-64/protobuf/v3.0.2-1/bin/protoc` -> `-DPROTOBUF_PROTOC_EXECUTABLE=/home/local/alice/sw/slc7_x86-64/protobuf/latest/bin/protoc`
7. In CLion, open Settings and go to the tab "Build,Execution,Deployment". 
8. Click `CMake` and expand the "CMake options". Copy the string you substituted.
9. Save
10. Reopen clion from a terminal in which you will have loaded the environment : `alienv load O2/latest`

## Tips

### Code formatting

1. [Download](https://github.com/AliceO2Group/CodingGuidelines/raw/master/settings-codestyle-clion.jar) the file
2. Go to File -> Import Settings and import the formatting settings.


