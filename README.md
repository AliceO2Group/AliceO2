
AliceO2
=======

Alice O2 project software. Simulation and reconstruction software for
the ALICE experiment at CERN based on ALFA and the FairRoot software.

Before compiling and installing AliceO2, the ALFA software must be
installed by choosing either the full or the minimum installation.

### Documentation
The documentation single entry point is [here](https://alice-o2.web.cern.ch/). 

### Installation 
In order to install with aliBuild you can follow [this tutorial](http://alisw.github.io/alibuild/o2-tutorial.html).

### Issue tracking system
We use JIRA to track issues. Head [here](https://alice.its.cern.ch/jira) to create tickets.

### Coding guidelines
The Coding Guidelines are [here](https://github.com/AliceO2Group/CodingGuidelines).

### Doxygen 
Turn it on in cmake (`cmake -DBUILD_DOXYGEN=ON ...`) before building (`make`) and then open 
`docs/doxygen/doc/html/index.html` from the build directory.

Doxygen documentation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Build system and directory structure
The build system and directory structure are described in 
[docs/doxygen/CMakeInstructions.md](@ref CodeOrganizationAndBuild).