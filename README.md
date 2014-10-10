AliceO2
=======

Alice O2 project software. Simulation and reconstraction software for Alice experiment at CERN base on ALFA and FairRoot software. 

### Step by Step installation


1. Install [FairSoft](https://github.com/FairRootGroup/FairSoft/tree/dev)

    we use here "fair_install" as a directory name, you can use what you went! 
    ```bash
    mkdir ~/fair_install
    cd ~/fair_install
    #git clone https://github.com/FairRootGroup/FairSoft.git
    git clone -b dev https://github.com/FairRootGroup/FairSoft.git
    cd FairSoft
    ./configure.sh
    # 1) gcc (on Linux) 5) Clang (on OSX)
    # 1) No Debug Info
    # 2) Internet (install G4 files from internet)
    # path: ~/fair_install/FairSoftInst
    ```
