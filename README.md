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
        mkdir build_o2
        cd build_o2
        cmake ../
        make
        . config.sh [or source config.csh]
