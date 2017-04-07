
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
`make doc` will generate the doxygen documentation.
To access the resulting documentation, open doc/html/index.html in your
build directory. To install the documentation when calling `make install`
turn on the variable `DOC_INSTALL`.

Doxygen documentation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Build system (cmake) and directory structure
The code organisation is described [here](docs/CodeOrganization.md).
The build system (cmake) is described [here](docs/CMakeInstructions.md).

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

### Documentation

The documentation single entry point is [here](https://alice-o2.web.cern.ch/).

### Issue tracking system

We use JIRA to track issues. Head [here](https://alice.its.cern.ch/jira) to create tickets.
