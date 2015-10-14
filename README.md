AliceO2
=======

Alice O2 project software. Simulation and reconstraction software for the ALICE experiment at CERN based on ALFA and the FairRoot software.

### Step by Step installation

1. Install [FairSoft/AlFa](https://github.com/FairRootGroup/FairSoft/tree/dev)

    we use here "alfa_src" as a directory name, you can change it to what ever you like

        git clone  https://github.com/FairRootGroup/FairSoft.git  alfa_src
        cd  alfa_src
        ./alfaconfig.sh
        # 1) gcc (on Linux) 5) Clang (on OSX)
        # 1) No Debug Info
        # 2) Internet (install G4 files from internet)
        # path: ~/AlFa

    To run the tests do:

        cd alfa_src/FairRoot/build_for_alfa/
        make test

2. Set several required shell variables, needed during the installation and running of the
   different software packages. Put these in your shell's rc file (~/.bashrc or ~/.cshrc).
   For bash:

        export SIMPATH=~/AlFa
        export FAIRROOTPATH=$SIMPATH/FairRoot

    or for csh:

        setenv SIMPATH ~/AlFa
        setenv FAIRROOTPATH $SIMPATH/FairRoot

3. Install the [AliceO2] (https://github.com/AliceO2Group/AliceO2) software

        git clone  https://github.com/AliceO2Group/AliceO2.git
        cd AliceO2
        mkdir build_o2
        cd build_o2
        cmake ../
        # -DBUILD_DOXYGEN=ON   ( add this option to cmake to generate the doxygen documentation)
        make
        . config.sh [or source config.csh]

### Generating the doxygen documentation

To automatically generate documentation for the AliceO2 project using Doxygen, set the flag -DBUILD_DOXYGEN=ON when calling cmake; the doxygen documentation will then be generated when calling make.  The generated html files can be found in the "doxygen/doc/html" subdirectory of the build directory.

Doxygen documantation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Compiling with custom DDS location

To include custom DDS location in the compilation, provide DDS_PATH flag when calling cmake. For example:
```bash
cmake -DDDS_PATH="/home/username/DDS/0.11.27.g79f48d4/" ..
```
