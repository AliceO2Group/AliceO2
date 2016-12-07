
AliceO2
=======

Alice O2 project software. Simulation, reconstruction and common software for
the ALICE experiment at CERN based on ALFA and the FairRoot software.

### Documentation
The documentation single entry point is [here](https://alice-o2.web.cern.ch/).

### Installation
In order to install with aliBuild you can follow [this tutorial](http://alisw.github.io/alibuild/o2-tutorial.html).

### Issue tracking system
We use JIRA to track issues. Head [here](https://alice.its.cern.ch/jira) to create tickets.

### Coding guidelines
The Coding Guidelines are [here](https://github.com/AliceO2Group/CodingGuidelines).
See [below](###Formatting) how to format your code accordingly.

### Doxygen
Turn it on in cmake (`cmake -DBUILD_DOXYGEN=ON ...`) before building (`make`) and then open
`docs/doxygen/doc/html/index.html` from the build directory.

Doxygen documentation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Build system and directory structure
The build system and directory structure are described in
[docs/doxygen/CMakeInstructions.md](@ref CodeOrganizationAndBuild).

### Formatting
##### Using clang-format
```bash
# Get the configuration file
git clone https://github.com/AliceO2Group/CodingGuidelines.git
cp CodingGuidelines/_clang-format-4  /path/to/O2/top/dir # (use _clang-format-3 if you use clang-format v3)

# Check the style
# Here any tag "<replacement " indicates a problem ("<replacements " with **s** is fine!)
clang-format -style=file -output-replacements-xml SOURCEFILE`
# shows what would the file content be after the reformatting
clang-format -style=file SOURCEFILE

# Apply the style to the file
clang-format -style=file -i SOURCEFILE
```

##### Using an IDE
A number of config files are available [here](https://github.com/AliceO2Group/CodingGuidelines) for various IDEs.
