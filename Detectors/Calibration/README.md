<!-- doxy
\page refDetectorsCalibration Module 'Detectors/Calibration'
/doxy -->

# Calibration flow for O2

The calibration flow of O2 foresees that every calibration device (expected to all run on the same EPN) will receive the TimeFrames from every EPN in an asynchronous way. The calibration device will have to process the TFs in time intervals (TimeSlots) which allow to create CCDB entries with the needed granularity and update frequency (defined by the calibration device itself).

## TimeSlotCalibration<Input, Container>
Each calibration device (to be run in a workflow) has to derive from `o2::calibration::TimeSlotCalibration`, which is a templated class that takes as types the Input type (i.e. the object to be processed, coming from the upstream device) and the Container type (i.e. the object that will contain the calibration data per TimeSlot). Each calibration device has to be configured with the following parameters:

```cpp
tf-per-slot : default length of a TiemSlot in TFs (will be widened in case of too little statistics)
max-delay   : maximum arrival delay of a TF with respect to the most recent one processed; if beyond this, the TF will be considered too old, and discarded
```

Each calibration device has to implement the following methods:

`initOutput`: initialization of the output object (typically a vector of calibration objects and another one with the associated CcdbObjectInfo;
`hasEnoughData` : method to determine whether a TimeSlot has enough data to be calibrated; if not, it will be merged to the following (in time) one;
`finalizeSlot` : method to process the calibration data accumulated in each TimeSlot;
`emplaceNewSlot` : method to creata a new TimeSlot; this is specific to the calibration procedure as it instantiates the detector-calibration-specific object.

## TimeSlot<Container>
The TimeSlot is a templated class which takes as input type the Container that will hold the calibration data needed to produce the calibration objects (histograms, vectors, array...). Each calibration device could implement its own Container, according to its needs.

The Container class needs to implement the following methods:

`fill`  : method to decide how to use the calibration data within the container (e.g. fill a vector);
`merge` : method to allow merging of the content of a TimeSlot to the content of the following one, when stastics is limited.
`print` : method to print the content of the Container

## detector-specific-calibrator-workflow

Each calibration will need to be implemented in the form of a workflow, whose options should include those for teh calibration device itself (`tf-per-slot` and `max-delay`, see above).
The output to be sent by the calibrator should include:

- a vector of the snapshots of the object to be put in the CCDB;
- a vector of the `o2::ccdb::CcdbObjectInfo` objects that contain the extra
information (metadata, startValidity...) associated to the objects themselves.

E.g.:

```c++
output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, i}, *image.get()); // vector<char>
output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, i}, w);               // root-serialized
    }
```
The origin of the output will always be `clbUtils::gDataOriginCLB`, while the description will be `clbUtils::gDataDescriptionCLBPayload` for the object itself, and `clbUtils::gDataDescriptionCLBInfo` for teh description.

## cccd-populator-workflow

This is the workflow that, connected to all workflows producting calibrations with different granularities and frequencies, will update the CCDB.
When adding a new calibration workflow, named e.g. "test-workflow", the following needs to be added to `ccdb-populator-workflow.cxx` in the `customize` method:

```cpp
policies.push_back(CompletionPolicyHelpers::defineByName("test-workflow.*", CompletionPolicy::CompletionOp::Consume));
```

The `--ccdb-path` option of the ccdb-populator-workflow allows to define the CCDB destination (e.g. `--ccdb-path localhost:8080.

