
ALICE O2 software 
=================

[![codecov](https://codecov.io/gh/AliceO2Group/AliceO2/branch/dev/graph/badge.svg)](https://codecov.io/gh/AliceO2Group/AliceO2/branches/dev)

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
The project uses `clang-format` to push for a common code formatting according
the the `clang-format` configuration files in this repository. With an adiabatic
approach, all changes have to follow the formatting rules. A tool script can be
used to integrate the formatting into `git` and suggest formatting only for
changed lines.

##### Install `clang-format` git integration
The `git-clang-format` Python script integrates `clang-format` into `git`.
Put it somewhere in your path and ensure that it is executable, e.g.
```bash
cd $HOME
mkdir -p bin
cd bin
wget //llvm.org/svn/llvm-project/cfe/trunk/tools/clang-format/git-clang-format
chmod u+x git-clang-format
```

Note: installation of the script will be added to build of AliceO2 software stack.

##### Checking formatting
Now, `git clang-format` will invoke `clang-format` on the changes in current files
or a specific commit. E.g. for the last commit
```
git clang-format HEAD~1
```

Or for all commits done with respect to the remote branch state
```
git clang-format origin/dev
```

##### Checking files
show updated version of complete file
```
clang-format -style=file SOURCEFILE
```

Directly apply the style to file
```
clang-format -style=file -i SOURCEFILE
```

### Using an IDE
A number of config files are available [here](https://github.com/AliceO2Group/CodingGuidelines) for various IDEs.
