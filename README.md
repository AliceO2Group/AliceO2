# ALICE O2 software {#mainpage}

<!--  \cond EXCLUDE_FOR_DOXYGEN -->

[![codecov](https://codecov.io/gh/AliceO2Group/AliceO2/branch/dev/graph/badge.svg)](https://codecov.io/gh/AliceO2Group/AliceO2/branches/dev)
[![JIRA](https://img.shields.io/badge/JIRA-Report%20issue-blue.svg)](https://alice.its.cern.ch/jira/secure/CreateIssue.jspa?pid=11201&issuetype=1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1493334.svg)](https://doi.org/10.5281/zenodo.1493334)

[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_O2_o2.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_O2_o2/fullLog.txt)
[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_o2_macos.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_o2_macos/fullLog.txt)
[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_o2checkcode_o2.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_o2checkcode_o2/fullLog.txt)
[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_O2_o2-dev-fairroot.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_O2_o2-dev-fairroot/fullLog.txt)

<!--  \endcond  -->

### Scope

The ALICE O2 software repository contains the framework, as well as the detector specific, code for the reconstruction, calibration and simulation for the ALICE experiment at CERN for Run 3 and 4. It also encompasses the commonalities such as the data format, and the global algorithms like the global tracking.
Other repositories in AliceO2Group contain a number of large common modules, for instance for Monitoring or Configuration.

### Website

The main entry point for O2 information is [here](http://alice-o2.web.cern.ch/).
A quickstart page can be found under [https://aliceo2group.github.io/](https://aliceo2group.github.io/).

### Building / Installation

In order to build and install O2 with aliBuild you can follow [this tutorial](http://alisw.github.io/alibuild/o2-tutorial.html).

### Issue tracking system

We use JIRA to track issues. [Report a bug here](https://alice.its.cern.ch/jira/secure/CreateIssue.jspa?pid=11201&issuetype=1).
Add the JIRA issue key (e.g. `O2-XYZ`) to the PR title or in a commit message to have the PR/commit appear in the JIRA ticket.

### Coding guidelines

The Coding Guidelines are [here](https://github.com/AliceO2Group/CodingGuidelines).
See [below](###Formatting) how to format your code accordingly.

### Doxygen

Documentation pages: [https://aliceo2group.github.io/AliceO2/](https://aliceo2group.github.io/AliceO2/).

`cmake --build . --target doc` will generate the doxygen documentation.
To access the resulting documentation, open doc/html/index.html in your
build directory. To install the documentation when calling `cmake --build . -- install` (or `cmake --install` for CMake >= 3.15)
turn on the variable `DOC_INSTALL`.

The instruction how to add the documentation pages (README.md) are available [here](doc/DoxygenInstructions.md).

### Build system (cmake) and directory structure

The code organisation is described [here](doc/CodeOrganization.md).
The build system (cmake) is described [here](doc/CMakeInstructions.md).

### Formatting

Rules and instructions are available in the repository
[CodingGuidelines](https://github.com/AliceO2Group/CodingGuidelines).
