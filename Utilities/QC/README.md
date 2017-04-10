QC
=======

Quality Control prototype for ALICE O2.

# Prerequisites
0. Installed AliceO2 and DDS software.
1. Set the environment variable SIMPATH to your FairSoft installation directory.
2. Set the environment variable FAIRROOTPATH to your FairRoot installation directory.

It is a good practice to run config.sh script from AliceO2 build directory to set all others variables such as PATH etc.

# Overview
This is a merging prototype for AliceO2 project. It uses FairMQ framework to provide distributed environment.

Project consists of four modules:
## Producer - produces Quality Control objects
Required arguments:

	- TH1F: DDS topology property id, device id, TH1F option, object name, object title, buffer capacity, number of bins

	- TH2F: DDS topology property id, device id, TH2F option, object name, object title, buffer capacity, number of bins

	- TH3F: DDS topology property id, device id, TH3F option, object name, object title, buffer capacity, number of bins

	- THnF: DDS topology property id, device id, THnF option, object name, object title, buffer capacity, number of bins

	- TTree: DDS topology property id, device id, TTree option, object name, object title, buffer capacity, number of bins, number of branches, number of entries in each branch

where:

	- DDS topology property id: id of the topology property holding merger address (e.g. mergerAddr)
	- device id: id of the device (e.g. mergerAddr)
	- option: one of the option of object type to produce (TH1F, TH2F, TH3F, THnF or TTree)
	- object name: name of the produced objects (e.g. histogramName)
	- object title: title of the produced objects (e.g. histogramTitle)
	- buffer capacity: capacity of the outpu buffer (e.g. 100)
	- number of bins: number of bins in produced QC data object (e.g. 1000)
	- number of branches: number of branches in TTree QC object (e.g. 4)
	- number of entries in each branch: number of entries in each branch in TTree QC object (e.g. 1000)

Run example for histogram:
```bash
runQCProducerDevice mergerAddr deviceID TH1F histogramName histogramTitle 100 1000
```

## Merger - merges received objects.
Required arguments:

	- DDS topology property id: id of the topology property holding merger address (e.g. mergerAddr)
	- device id: id of the device (e.g. deviceID)
	- required number of objects with the same name to merge (e.g. 100)
	- merger input TCP port (e.g. 5016)
	- input buffer capacity (e.g. 500000)
	- output address with TCP port number (e.g. tcp://login01.pro.cyfronet.pl:5004)

Run example:
```bash
runQCMergerDevice mergerAddr deviceID 100 5016 500000 tcp://login01.pro.cyfronet.pl:5004
```
## Viewer - provides visualization of merged objects.
Optional arguments:

	- drawing option: drawing option passed to Draw function of a QC object (e.g. branchtoDrawName)

Run example:
```bash
runQCViewerDevice branchToDrawName
```
## MetricsExtractor - used for metrics extraction from nodes.
Sends DDS custom commands to all of the nodes in a topology. It accepts responses as a json structures with valid custom command name.

Required arguments:

	- output file suffix name: suffic to be added to out file name of nodes metrics (e.g. metricSuffix)

Run example:
```bash
runQCMetricsExtractor metricSuffix
```

# Compile software
1. Go to build folder of AliceO2 software
2. cmake ../
3. cd Utilities/QA
4. make all

# Unit tests
All modules are provided with unit tests written in BOOST test framework. Each module has tests in "Tests" subdirectory.
To run all unit tests type ```ctest ```

# Run system
See this page: http://dds.gsi.de/doc/nightly/RMS-plugins.html#slurm-plugin to execute system with DDS SLURM plug-in.

Mergers and Producers have to be run with DDS topology. MetricsExtractor and Viewer should be run with bash shell.

## DDS topologies examples
1. 2 peoducers and 1 merger
```xml
<topology id="QA">

    <var id="noOfProducers" value="2" />

    <property id="merger1Addr" />

    <decltask id="Producer1">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCProducerDevice merger1Addr deviceID TH1F histogramName histogramTitle 4 100</exe>
        <properties>
          <id access="read">merger1Addr</id>
        </properties>
    </decltask>

    <decltask id="Merger1">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCMergerDevice merger1Addr Merger1 100 5015 500000 tcp://login01.pro.cyfronet.pl:5004</exe>
        <properties>
          <id access="write">merger1Addr</id>
        </properties>
    </decltask>

    <declcollection id="producers1">
      <tasks>
         <id>Producer1</id>
      </tasks>
   </declcollection>

    <declcollection id="mergers1">
      <tasks>
         <id>Merger1</id>
      </tasks>
   </declcollection>

    <main id="main">
        <group id="producersGroup1" n="${noOfProducers}">
            <collection>producers1</collection>
        </group>
        <group id="mergersGroup1" n="1">
            <collection>mergers1</collection>
        </group>
    </main>

</topology>

```


2. 500 producers and 2 mergers
```xml
<topology id="QA">

    <var id="noOfProducers" value="250" />

    <property id="merger1Addr" />
    <property id="merger2Addr" />

    <decltask id="Producer1">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCProducerDevice merger1Addr deviceID TH1F histogramName histogramTitle 4 100</exe>
        <properties>
          <id access="read">merger1Addr</id>
        </properties>
    </decltask>

    <decltask id="Producer2">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCProducerDevice merger2Addr deviceID TH1F histogramName histogramTitle 4 100</exe>
        <properties>
          <id access="read">merger2Addr</id>
        </properties>
    </decltask>

    <decltask id="Merger1">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCMergerDevice merger1Addr Merger1 250 5015 500000 tcp://login01.pro.cyfronet.pl:5004</exe>
        <properties>
          <id access="write">merger1Addr</id>
        </properties>
    </decltask>

    <decltask id="Merger2">
        <exe reachable="false">@CMAKE_BINARY_DIR@/runQCMergerDevice merger2Addr Merger2 250 5016 500000 tcp://login01.pro.cyfronet.pl:5004</exe>
        <properties>
          <id access="write">merger2Addr</id>
        </properties>
    </decltask>

    <declcollection id="producers1">
      <tasks>
         <id>Producer1</id>
      </tasks>
   </declcollection>

    <declcollection id="producers2">
      <tasks>
         <id>Producer2</id>
      </tasks>
   </declcollection>

    <declcollection id="mergers1">
      <tasks>
         <id>Merger1</id>
      </tasks>
   </declcollection>

    <declcollection id="mergers2">
      <tasks>
         <id>Merger2</id>
      </tasks>
   </declcollection>

    <main id="main">
        <group id="producersGroup1" n="${noOfProducers}">
            <collection>producers1</collection>
        </group>
		<group id="producersGroup2" n="${noOfProducers}">
            <collection>producers2</collection>
        </group>
        <group id="mergersGroup1" n="1">
            <collection>mergers1</collection>
        </group>
		 <group id="mergersGroup2" n="1">
            <collection>mergers2</collection>
        </group>
    </main>

</topology>

```
## How to run topology with DDS SLURM plug-in
This is an example of running first topology from previous examples:
```
dds-server start -s
dds-submit -r slurm -n 3 slurm.cfg
dds-topology --set @PATH_TO_TOPOLOGY_FILE@/topology.xml
dds-topology --activate
```
