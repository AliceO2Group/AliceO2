
AliceO2
=======

Alice O2 project software. Simulation and reconstraction software for the ALICE experiment at CERN based on ALFA and the FairRoot software.

### Installation of Alfasoft (FairSoft)
Before start installing please be sure that your system has all the required libraries (see [FairSoft/DEPENDENCIES](https://github.com/FairRootGroup/FairSoft/blob/master/DEPENDENCIES)) for details.

#### Full installation:
The full installation will install all packages on [FairSoft](https://github.com/FairRootGroup/FairSoft/tree/dev) + [DDS](https://github.com/FairRootGroup/DDS) + [FairRoot](https://github.com/FairRootGroup/FairRoot/tree/dev) and [AliROOT]() 

This installation:
* Need a fast network connection 
* will take the __development__ branches for all above packages.
* It will install Geant4 Data files 
* Need about __10 GByte__ of disk space (8.1 for the source and objects files, etc and 2.2 GByte for the installation)

##### Step by step installation
1. Install [FairSoft/AlFa](https://github.com/FairRootGroup/FairSoft/tree/dev)

we use here "alfa_src" as a directory name, you can change it to what ever you like

```bash 
git clone  https://github.com/FairRootGroup/FairSoft.git  alfa_src
cd  alfa_src
./alfaconfig.sh
# 1) gcc (on Linux) 5) Clang (on OSX)
# 1) No Debug Info
# 2) Internet (install G4 files from internet)
# path: ~/AlFa
```

To run the tests do:
```bash 
cd alfa_src/FairRoot/build_for_alfa/
make test
```
2. Set several required shell variables, needed during the installation and running of the
different software packages. Put these in your shell's rc file (~/.bashrc or ~/.cshrc).
For bash:
```bash 
export SIMPATH=~/AlFa
export FAIRROOTPATH=$SIMPATH/FairRoot
```
or for csh:
```bash 
setenv SIMPATH ~/AlFa
setenv FAIRROOTPATH $SIMPATH/FairRoot
```

#### Minumum installtion (reconstruction only installation)
This installaiton will exclude:
1. Simulation engines (Geant3/4)
2. Event generators (Pythia6/8)
3. VGM, VMC
##### Step by step installation
Edit the recoonly file in alfa_src, and set your compiler and installation directory.
(if you went to use ROOT 6 switch it on!)
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
./configure.sh  recoonly
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

If you choosed the minimum installation for ALFA before (in step one above) AliceO2 will not include the simulation and reconstruction packages.

Set the variable SIMPATH to your FairSoft/alfasoft installation directory

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

