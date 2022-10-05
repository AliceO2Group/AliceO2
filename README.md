# DigitClusterWorkflow

## Allow to read from stream or file

### Contents : 

- include/HMPIDWorkflow
  - _DigitsToClustersSpec.h_ : Spec for digits-to-cluster-Workflow
- src
  - _ClustersToRootSpec2.h_ MakeRootTreeWriterSpec: writes cluster-out-file
  - _DigitsToClustersSpec.cxx_ : Spec for digits-to-cluster-Workflow
  - _digits-to-clusters-workflow.cxx_ : executable for digits-to-clusters


#### Reads digitis upstream from real data by default.



### Command to create digits:
In the same folder where hits root file is, type the command:

    o2-sim-digitizer-workflow --onlyDet HMP

This creates a digits file named _"hmpiddigits.root"_ in the working folder. 
### Clusterization
    o2-hmpid-digits-to-clusters-workflow
    
If reading from the digit-file made in previous step is desired, use the option `--read-from-file` .  
Digits will by default be written upstream, but can be written to a root-file by the `--write-to-file` option.

The default input-file is named _"hmpiddigits.root"_, and is defined in [HMPIDDigitWriterSpec.h](https://github.com/AliceO2Group/AliceO2/blob/dev/Steer/DigitizerWorkflow/src/HMPIDDigitWriterSpec.h)  
The input-file can also be altered by passing the argument `--hmpid-digit-infile fileName.root`

If the clusters are written to file, the default file-name is _"hmpidclusters.root"_  
~~The file-name can be altered by passing the argument `--out-file fileName.root`~~ : This is not done yet


# Change all Legacy Physics classes for HMPID
https://root.cern.ch/doc/master/group__Physics.html

### Change TVector2 -> Vector2D and TVector3 -> Vector3D in HMPID::Recon



### [Trotation in HMPID-param](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/src/Param.cxx)
  * Does not seem to be used?
### [TLorentzVector in HMPID-detector](https://github.com/AliceO2Group/AliceO2/blob/54d91df6bd1f9008ed8caa748820cfc3b95535e4/Detectors/HMPID/simulation/src/Detector.cxx)
  * [ROOT::Math::LorentzVector](https://github.com/eflatlan/AliceRecon/blob/1483a2302205717d9c97272287090a46daf4a338/Detectors/HMPID/simulation/src/Detector.cxx#L189)


### [TVector3 in HMPID-param header](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/include/HMPIDBase/Param.h#L19)
  1. lors2Mars [old](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/include/HMPIDBase/Param.h#L191-L196) -> [new](https://github.com/eflatlan/AliceRecon/blob/1483a2302205717d9c97272287090a46daf4a338/Detectors/HMPID/base/include/HMPIDBase/Param.h#L227-L235) 
  2. norm [old](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/include/HMPIDBase/Param.h#L213-L218) -> [new](https://github.com/eflatlan/AliceRecon/blob/1483a2302205717d9c97272287090a46daf4a338/Detectors/HMPID/base/include/HMPIDBase/Param.h#L255-L260) 
  
### [TVector3 in HMPID-param src](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/src/Param.cxx#L289)
  * sigma2 [old](https://github.com/AliceO2Group/AliceO2/blob/03608ff899d444d52571dbed14a0106ae4616562/Detectors/HMPID/base/src/Param.cxx#L289-L304) -> [new](https://github.com/eflatlan/AliceRecon/blob/1483a2302205717d9c97272287090a46daf4a338/Detectors/HMPID/base/src/Param.cxx#L300-L320)

# ALICE O2 software {#mainpage}

<!--  /// \cond EXCLUDE_FOR_DOXYGEN -->

[![codecov](https://codecov.io/gh/AliceO2Group/AliceO2/branch/dev/graph/badge.svg)](https://codecov.io/gh/AliceO2Group/AliceO2/branches/dev)
[![JIRA](https://img.shields.io/badge/JIRA-Report%20issue-blue.svg)](https://alice.its.cern.ch/jira/secure/CreateIssue.jspa?pid=11201&issuetype=1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1493334.svg)](https://doi.org/10.5281/zenodo.1493334)

[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_O2_o2.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_O2_o2/fullLog.txt)
[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_o2_macos.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_o2_macos/fullLog.txt)
[![](http://ali-ci.cern.ch/repo/buildstatus/AliceO2Group/AliceO2/dev/build_o2checkcode_o2.svg)](https://ali-ci.cern.ch/repo/logs/AliceO2Group/AliceO2/dev/latest/build_o2checkcode_o2/fullLog.txt)

<!--  /// \endcond  -->

### Scope

The ALICE O2 software repository contains the framework, as well as the detector specific, code for the reconstruction, calibration and simulation for the ALICE experiment at CERN for Run 3 and 4. It also encompasses the commonalities such as the data format, and the global algorithms like the global tracking.
Other repositories in AliceO2Group contain a number of large common modules, for instance for Monitoring or Configuration.

### Website

The main entry point for O2 information is [here](https://alice-o2-project.web.cern.ch).
A quickstart page can be found under [https://aliceo2group.github.io/](https://aliceo2group.github.io/).

### Building / Installation

In order to build and install O2 with aliBuild you can follow [this tutorial](http://alisw.github.io/alibuild/o2-tutorial.html).

### Discussion Forum

Users can ask for support in [ALICE Talk](https://alice-talk.web.cern.ch).

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

The instruction how to add the documentation pages (README.md) are available [here](https://github.com/AliceO2Group/AliceO2/blob/dev/doc/DoxygenInstructions.md).

### Build system (cmake) and directory structure

The code organisation is described [here](https://github.com/AliceO2Group/AliceO2/blob/dev/doc/CodeOrganization.md).
The build system (cmake) is described [here](https://github.com/AliceO2Group/AliceO2/blob/dev/doc/CMakeInstructions.md).

### Formatting

Rules and instructions are available in the repository
[CodingGuidelines](https://github.com/AliceO2Group/CodingGuidelines).

### Enable C++ compiler warnings

Currently O2 is built with minimal compiler warnings enabled. This is going to change in the near future. In the transition period, developers have to manualy enable warnings by building O2 with `ALIBUILD_O2_WARNINGS` environment variable set e.g. using the `-e`  option of `alibuild` e.g: 
```bash 
aliBuild build --debug -e ALIBUILD_O2_WARNINGS=1 --defaults o2 O2
``` 
A helper script that extracts warnings from the build log skipping duplicates is available [here](https://github.com/AliceO2Group/AliceO2/blob/dev/scripts/filter-warnings.sh)
