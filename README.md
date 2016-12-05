
AliceO2
=======

Alice O2 project software. Simulation and reconstruction software for
the ALICE experiment at CERN based on ALFA and the FairRoot software.

Before compiling and installing AliceO2, the ALFA software must be
installed by choosing either the full or the minimum installation.

### Documentation
The documentation single entry point is [here](https://alice-o2.web.cern.ch/).

### Issue tracking system
We use JIRA to track issues. Head [here](https://alice.its.cern.ch/jira) to create tickets.

### Coding guidelines
The Coding Guidelines are [here] (https://github.com/AliceO2Group/CodingGuidelines).

### Installation via aliBuild

In order to install with aliBuild you can follow the tutorial at:

    http://alisw.github.io/alibuild/o2-tutorial.html

### Installation with ALFA (FairSoft)

Please be sure that your system has all
the required libraries (as listed on
[FairSoft/DEPENDENCIES](https://github.com/FairRootGroup/FairSoft/blob/master/DEPENDENCIES)).

#### Full installation:

The full installation will install [FairSoft](https://github.com/FairRootGroup/FairSoft/tree/dev), [DDS](https://github.com/FairRootGroup/DDS), [FairRoot](https://github.com/FairRootGroup/FairRoot/tree/dev)

The installation:
* Needs a fast network connection
* Will take the __development__ branches for all the above packages.
* Will install the Geant4 Data files
* Needs about __10 GBytes__ of disk space (8.1 for the source and objects files, etc and 2.2 GBytes for the installation)

##### Step by step instructions for the full installation
1. Install [FairSoft](https://github.com/FairRootGroup/FairSoft/tree/dev)
"alfa_src" is referred as the directory where the ALFA sources exist, you can specify an alternative name if you wish
```bash
git clone  https://github.com/FairRootGroup/FairSoft.git  alfa_src
cd  alfa_src
./configure.sh
# 1) gcc (on Linux) 5) Clang (on OSX)
# 1) No Debug Info
# 2) Internet (install G4 files from internet)
# path: ~/AlFa
```
2. Install [DDS](https://github.com/FairRootGroup/DDS)
```bash
git clone https://github.com/FairRootGroup/DDS
cd DDS
mkdir build
cd build
BOOST_ROOT=$ALFA_installation_dir cmake -DCMAKE_INSTALL_PREFIX="DDS_install_prefix"  -C ../BuildSetup.cmake ../
```
3. Install [FairRoot](http://fairroot.gsi.de/?q=node/82)  

```bash
#Set the shell variable SIMPATH to the installation directory
export SIMPATH= ALFA_installation_dir
[setenv SIMPATH ALFA_installation_dir]

git clone -b dev https://github.com/FairRootGroup/FairRoot.git
cd FairRoot
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="FairRoot_installation_dir" -DDDS_PATH="DDS_install_prefix"  ..
make
make install
```
To run the tests do:

```bash
# To run test: make new shell, do not define SIMPATH
cd FairRoot/build
make test
```
To run the tests do:
```bash
cd alfa_src/FairRoot/build_for_alfa/
make test
```

#### Minimum installation (reconstruction only installation)
This installation will exclude:
1. Simulation engines (Geant3/4)
2. Event generators (Pythia6/8)
3. VGM, VMC
##### Step by step for the minimum installation
Edit the "[recoonly.conf](https://github.com/FairRootGroup/FairSoft/blob/master/recoonly.conf)" file in alfa_src, and set your compiler and installation directory.
(the use of ROOT 6 can be also specified here if needed!)
```bash
compiler= <your compiler>
debug=yes
optimize=no
geant4_download_install_data_automatic=no
geant4_install_data_from_dir=no
build_root6=no
build_python=no
install_sim=no
SIMPATH_INSTALL= <ALFA_installation_dir>
```

1. Install FairSoft
```bash
git clone  https://github.com/FairRootGroup/FairSoft.git  alfa_src
cd  alfa_src
./configure.sh  recoonly.conf
```
2. Install [FairRoot](http://fairroot.gsi.de/?q=node/82)

```bash
# Set the shell variable SIMPATH to the installation directory
export SIMPATH= ALFA_installation_dir
[setenv SIMPATH ALFA_installation_dir]

git clone -b dev https://github.com/FairRootGroup/FairRoot.git
cd FairRoot
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="FairRoot_installation_dir" ..
make
make install
```
To run the tests do:

```bash
# To run test: make new shell, do not define SIMPATH
cd FairRoot/build
make test
```

### Install the [AliceO2](https://github.com/AliceO2Group/AliceO2) software

If ALFA was built using the minimum installation AliceO2 will not include the simulation and reconstruction packages.

Set the variable SIMPATH to your ALFA/FairSoft installation directory

```bash
export SIMPATH=ALFA_installation_dir
export FAIRROOTPATH=FairRoot_installation_dir
```

```bash
git clone  https://github.com/AliceO2Group/AliceO2.git
cd AliceO2
mkdir build_o2
cd build_o2
cmake ../
# -DBUILD_DOXYGEN=ON   ( add this option to cmake to generate the doxygen documentation)
make
. config.sh [or source config.csh]
```

### Generating the doxygen documentation

To automatically generate documentation for the AliceO2 project using Doxygen, set the flag -DBUILD_DOXYGEN=ON when calling cmake; the doxygen documentation will then be generated when calling make.  The generated html files can be found in the "doxygen/doc/html" subdirectory of the build directory.

Doxygen documantation is also available online [here](http://aliceo2group.github.io/AliceO2/)

### Compiling with custom DDS location

To include custom DDS location in the compilation, provide DDS_PATH flag when calling cmake. For example:
```bash
cmake -DDDS_PATH="/home/username/DDS/0.11.27.g79f48d4/" ..
```
