Hough Transform
===============

This is the groundwork for the Hough Transform algorithm implementation. The runHough executable takes as an argument an event number (i.e. runHough 032) and for the given event it loads all clusters from the corresponding data files.

### Step by Step installation and execution

1. The runHough executable depends on the AliRoot HLT libraries. For this, an optional dependency to AliRoot has been added to the AliceO2 framework. To build the executable, the path to the AliRoot installation must be given at configuration time. For example:

    cmake -DCMAKE_INSTALL_PREFIX:PATH=.. -DCMAKE_CXX_FLAGS="-std=c++11" .. -DALIROOT="/opt/alice/external/AliRoot"

It is important that AliRoot, FairRoot and AliceO2 have been built against the same version of ROOT. To ensure that the prerequisite packages were compiled and installed correctly, the alfaconfig.sh script that is included as part of the FairSoft installation can be used.

2. Raw data files should be retrieved from AliEn for a given run. The implementation can be tested using the raw files corresponding to run 167808:

    alien-token-init <username>
    aliensh

    cd /alice/data/2011/LHC11h/000167808/raw/
    cp 11000167808000.10.root file://tmp
    exit

    mv /tmp/11000167808000.10.root raw.root

Then, the necessary scripts to perform the clusterization for the data should be coppied to the current directory:

    cp ${AliRoot}/HLT/exa/recraw-local.C .
    cp ${AliRoot}/HLT/exa/EnableHLTInGRP.C .
    cp ${AliceO2}devices/aliceHLTwrapper/macros/hltConfigurations.C .

Finally, the following commands should executed:

    aliroot -b -q -l hltConfigurations.C recraw-local.C'("raw.root", "raw://", 0, 0, "HLT TPC", "loglevel=0x79 chains=cluster-collection", "local://./OCDB")'

    aliroot -b -q -l EnableHLTInGRP.C'(167808, "local://./OCDB", "local://./OCDB")'
    rm galice.root QA.root
    aliroot -b -q -l hltConfigurations.C recraw-local.C'("raw.root", "local://OCDB", -1, -1, "HLT", "loglevel=0x79 chains=cluster-collection")' 2>&1 | tee cluster-collection.log

The result should be a directory, called "emulated-tpc-clusters" that will be the input of runHough. By executing

    runHough 032

the executable will load all the clusters from the emulated-tpc-clusters/event032 subdirectory. After the execution, a graphics file "clusters.pdf" will be created in the current directory depicting the coordinates of the loaded clusters.
