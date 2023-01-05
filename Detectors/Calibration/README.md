<!-- doxy
\page refDetectorsCalibration Detectors Calibration
/doxy -->

# Time-interval based calibration flow for O2

The calibration flow of O2 foresees that all calibration devices are running on dedicated EPN calibration nodes. These nodes are also called aggregator nodes. A particular calibration device can only run on a single calibration node, as it is supposed to receive the complete input data from each EPN processing node.
The processing nodes can send the data either for every TF or sporadically. In the latter case for example accumulated for every 10 TFs.
In both cases the calibration input reaches the calibration nodes asynchronously. And since there are up to 250 EPNs processing TFs in two different NUMA domains, TFs which reach the calibration node consecutively might in reality be 500 TFs apart in absolute time.
This is because the processing time of individual TFs can vary quite significantly.
The calibration devices have to prepare the calibration objects for time intervals (TimeSlots) which are defined by the user.
They have a specificied duration and a minimum statistics requirement.
For example a calibration is supposed to aggregate data for 10 minutes and for a meaningful calibration within these 10 minutes a certain amount of global tracks is required. In case after 10 minutes the amount of global tracks is not sufficient, the framework can automatically increase the interval of 10 minutes until enough global tracks are available to create the calibration object.


## TimeSlotCalibration<Container>
Each calibration device which is supposed to run on the aggregator should derive from `o2::calibration::TimeSlotCalibration<Container>`. It is a templated class. The `Container` type is the object in which the calibration data per TimeSlot will be accumulated.

### Configuration of the TimeSlot

Internally the default length of a TimeSlot is calculated in number of TFs. The default TF length is 128 orbits, but theoretically this can change. Therefore it is advised to set the TimeSlot length via the methods `setSlotLengthInSeconds(int s)` or `setSlotLengthInOrbits(int n)`, which will be internally converted (and rounded) to the corresponding number of TFs at the 1st TF processing. At that time we know the TF length from the GRPECS object.
One can also set the number of TFs per slot directly via `setSlotLength(o2::calibration::TFType n)`. With `setSlotLength(o2::calibration::INFINITE_TF)` there will be only 1 slot at a time, valid till infinity. A special mode is configured with `setSlotLength(0)` in which case there will only be a single slot, w/o explicit boundaries which is filled until the statistics is reached. In case `setSlotLength(0)` is configured we can also use `setCheckIntervalInfiniteSlot(o2::calibration::TFType updateInterval)` in which case the calibration checks whether the statistics is sufficient only after `updateInterval` TFs. Otherwise it would check after every TF.

The TFs arrive asynchronously at the aggregator node. The `TimeSlotCalibration` keeps a `std::deque` of TimeSlots for which it aggregates the input data simultaneously. Whenever a slot has reached its configured duration the statistics requirement is checked. In case it is not fulfilled, the slot can be extended or merged to the previous slot in order to obtain the required statistics.

By default, TFs which arrive more than `3 * SlotLengthInTF` later than the most recent TF processed are discarded. This maximum delay can be configured via `setMaxSlotsDelay(int nSlots)`. If it is set to 4 and each slot has the length of 30 TFs, then upon processing of TF 121 the input from TF0 would be discarded, if it was not already processed.


In order to prepare only one CCDB object at the end of the run you can use `setUpdateAtTheEndOfRunOnly()`. In this case all the above settings for the slot duration are irrelevant. And upon the `endOfStream` of your calibration device you should make a call to `checkSlotsToFinalize()`.


### Mandatory methods to implement when deriving from `o2::calibration::TimeSlotCalibration<Container>`


- `void initOutput()`: initialization of the output object (typically a vector of calibration objects and another one with the associated CcdbObjectInfo);

- `bool hasEnoughData(const o2::calibration::TimeSlot<Container>& slot)` : method to determine whether a TimeSlot has enough data to be calibrated; if not, it will be merged to the following (in time) one;

- `void finalizeSlot(o2::calibration::TimeSlot<Container>& slot)` : method to process the calibration data accumulated in each TimeSlot;

- `o2::calibration::TimeSlot<Container>& slot emplaceNewSlot(bool front, TFType tstart, TFType tend)` : method to creata a new TimeSlot; this is specific to the calibration procedure as it instantiates the detector-calibration-specific object.

See e.g. LHCClockCalibrator.h/cxx in AliceO2/Detectors/TOF/calibration/include/TOFCalibration/LHCClockCalibrator.h and  AliceO2/Detectors/TOF/calibration/srcLHCClockCalibrator.cxx

## TimeSlot<Container>

The TimeSlot is a templated class which takes as input type the Container that will hold the calibration data needed to produce the calibration objects (histograms, vectors, array...). Each calibration device could implement its own Container, according to its needs.

The Container class needs to implement the following methods:

- `void merge(const Container* prev)` : method to allow merging of the content of a TimeSlot to the content of the following one, when stastics is limited.

- `void print()` : method to print the content of the Container

- `void fill(DATA data, ...)`  : method to decide how to use the calibration data within the container (e.g. fill a vector). The type of `DATA` is usually `const gsl::span<your-input-data>`, but can also be anything else. Optionally the `fill` method can accept additional input of arbitrary type;

or, alternatively

- `void fill(o2::dataformats::TFIDInfo& ti, DATA data, ...)`  : method to decide how to use the calibration data within the container (e.g. fill a vector) and having access to the TFIDInfo struct providing relevant info for current TF (tfCounter, runNumber, creationTime etc.).
If provided, this latter method will be used.

See e.g. LHCClockCalibrator.h/cxx in AliceO2/Detectors/TOF/calibration/include/TOFCalibration/LHCClockCalibrator.h and  AliceO2/Detectors/TOF/calibration/srcLHCClockCalibrator.cxx

The Slot provides a generic methods to access its boundaries: `getTFStart()` and `getTFEnd()` in terms of TF counter (as assigned by the DataDistribution) and `getStartTimeMS()`, `getEndTimeMS()` for the absolute time stamp in milleseconds.

## detector-specific-calibrator-workflow

Each calibration will need to be implemented in the form of a workflow, whose options should include those for the calibration device itself (for example the slot length and statistics requirement).
The output to be sent by the calibrator should include:

*   a vector of the snapshots of the object to be put in the CCDB;

*   a vector of the `o2::ccdb::CcdbObjectInfo` objects that contain the extra
information (metadata, startValidity...) associated to the objects themselves.

The origins of the pair of outputs will always be `o2::calibration::Utils::gDataOriginCDBPayload` and `o2::calibration::Utils::gDataOriginCDBWrapper` respectively, while the DataDescription must be unique for given calibration type, e.g.

```c++
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LHCphase", i}, *image.get()); // vector<char>
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LHCphase", i}, w);            // root-serialized
```

Note that in order to access the absolute time of the slot boundaries, one should subscribe to CTP orbit reset time object (at least) via GRPGeomHelper class.

See e.g. AliceO2/Detectors/TOF/calibration/testWorkflow/LHCClockCalibratorSpec.h,  AliceO2/Detectors/TOF/calibration/testWorkflow/lhc-clockphase-workflow.cxx

## Integration of calibration workflows into the global framework

For the synchronous processing on the EPN the calibration workflows are grouped according to their origin (BARREL, CALO, MUON and FORWARD) and to the nature of their input (TF for devices expecting input for every TF and SPORADIC for devices expecting input sporadically).
For each group (e.g. `BARREL_TF`) a pair of `o2-dpl-output-proxy` running on the processing EPNs and `o2-dpl-raw-proxy` running on the calibration nodes is initialized and these proxys are used to transfer the input from processing nodes to the calibration node.
Have a look at the `DATA/common/setenv_calib.sh` script in O2DPG where for each calibration the required data descriptors are added to the proxies.
In addition there is always some logic to decide whether a specific calibration should be enabled or not.

The workflow which is running on the processing nodes should be added in the `prodtests/full-system-test/calib-workflow.sh` script.
The workflow running on the aggregator should be added to `prodtests/full-system-test/aggregator-workflow.sh`.

## Calibrating over multiple runs

Some statistics-hungry calibrations define single time-slot which integrates data of the whole run. If there is a possibility that for the short run the slot will not accumulate enough statistics,
one can save the user-defined content of the slot to a file in the dedicated partition on the calibrator node
and adopt data from this file in the next run. In order to do that the calibrator class derived from the TimeSlotCalibration must:

* set fixed file name to write via `setSaveFileName(const std::string& n)` method. Also, the corresponding workflow should provide/parse an option to set the output directory.

* implement virtual method `bool saveLastSlotData(TFile& fl)` which writes content of the (last and only) slot into the provided file handler. It is up to detector to define the format of the stored data. The framework will write to the same file a
TimeSlotMetaData struct describing the start/end timestamps and start/end runs for the data written.

* implement virtual method `bool adoptSavedData(const TimeSlotMetaData& metadata, TFile& fl)` which reads and adds saved data to the slot in the new run. Provided metadata should be used to judge if the saved data are useful.

* decide e.g. in the finalizeSlot method if the slot content must be saved to be accounted in the following run and call `saveLastSlot()` in that case.

* in the beginning of the processing, e.g. after the 1st call of the `process(..)` method (where the time-slot will be created) call `loadSavedSlot()` method, i.e.
  ```
  auto data = pc.inputs().get<...>; // get input data
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
  mCalibrator->process(data);
  static bool firstCall = true;
  if (firstCall && getNSlots()) {
    firstCall = false;
    loadSavedSlot();
  }
  ```
  Make sure the static method `o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());` was called from the `run()` method before the `process(...)` call.

The slot saving and loading will be done only if `setSavedSlotAllowed(true)` was called explicitly from the calibrator device before the processing starts (e.g. in the `init()` method).
Since one can have multiple instances of the calibrator device
running at the same time (in staging and produnction partitions, synthetic and real runs) it is important to make sure that this method is called only for the physics run calibration.

In order to not pollute calibration node disk, the file will be removed in the end of `loadSavedSlot()` call.

## ccdb-populator-workflow

This is the workflow that, connected to all workflows producting calibrations with different granularities and frequencies, will update the CCDB.

The `--ccdb-path` option of the ccdb-populator-workflow allows to define the CCDB destination (e.g. `--ccdb-path localhost:8080`).

By default the `ccdb-populator-workflow` will upload to the CCDB the inputs with any `SubSpec`. There is a possibility to run with multiple `ccdb-populators`, e.g. each one writing to particular CCDB server
(this might be needed e.g. to populate the production CCDB and the transient CCDB for DCS exchange, with the same or different objects). This can be done by piping
two populators, one of which should have device name modified. Additionally, one can pass an option for each populator to process only inputs with specific `SubSpecs`.
E.g. if your calibration device sends to the output
```
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ObjA", 0}, *imageA.get());
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ObjA", 0}, wA);
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ObjB", 1}, *imageB.get());
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ObjB", 1}, wB);
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ObjC", 2}, *imageC.get());
output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ObjC", 2}, wC);
```
in the workflow defined as:
```
<your_workflow> | o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" --name-extention loc  --sspec-min 1 --sspec-max 10 |
o2-calibration-ccdb-populator-workflow --sspec-min 0 --sspec-max 1  -b
```
then the `ObjA` will be uploaded only to the default server (`https://alice-ccdb.cern.ch`), `ObjB` will be uploaded to both default and `local` server and
`ObjC` will be uploaded to the `local` server only.

By default the ccdb-populator-workflow will not produce `fatal` on failed upload. To require it an option `--fatal-on-failure` can be used.

<!-- doxy
* \subpage refDetectorsCalibrationtestMacros
/doxy -->
