
ALICE O2 software 
=================

### Scope
The ALICE O2 software repository contains the framework, as well as the detector specific, code for the reconstruction, calibration and simulation for the ALICE experiment at CERN for Run 3 and 4. It also encompasses the commonalities such as the data format, and the global algorithms like the global tracking. 
Other repositories in AliceO2Group contain a number of large common modules, for instance for Monitoring or Configuration.

### Website
The main entry point for O2 information is [here](http://alice-o2.web.cern.ch/).
A quickstart page can be found under [https://aliceo2group.github.io/](https://aliceo2group.github.io/).

### Installation
In order to install with aliBuild you can follow [this tutorial](http://alisw.github.io/alibuild/o2-tutorial.html).

### Issue tracking system
We use JIRA to track issues. Head [here](https://alice.its.cern.ch/jira) to create tickets.

### Coding guidelines
The Coding Guidelines are [here](https://github.com/AliceO2Group/CodingGuidelines).
See [below](###Formatting) how to format your code accordingly.

### Doxygen
Documentation pages: [https://aliceo2group.github.io/AliceO2/](https://aliceo2group.github.io/AliceO2/).

`make doc` will generate the doxygen documentation.
To access the resulting documentation, open doc/html/index.html in your
build directory. To install the documentation when calling `make install`
turn on the variable `DOC_INSTALL`.

Doxygen documentation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Build system (cmake) and directory structure
The code organisation is described [here](doc/CodeOrganization.md).
The build system (cmake) is described [here](doc/CMakeInstructions.md).

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
