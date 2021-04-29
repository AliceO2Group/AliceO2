## FIT Calibration workflow

###Overview
The module was implemented in a generic way. Adding new calibration object should be easy and intuitive. Calibration device uses https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/Calibration for TimeFrame management (see link above for more detailed information).

###How to use TimeSlotCalibration?
####It is required to define classes listed below to be able to use prepared TimeSlotCalibration module.
- TimeSlot Container class (for current state of the code, example - `FT0ChannelTimeTimeSlotContainer`). Container is responsible for data storage (CFDTime, Charge, etc.) and providing data for calibration object generation
- Calibration Info Object class (for current state of the code, example - `FT0CalibrationInfoObject`). This class is responsible for providing input data which will be saved in TimeSlot Container instance. (so we can assume it's simplified channel data type, with extracted data that will be required for calibration process)
- Calibration Object class (for current state of the code, example - `FT0ChannelTimeCalibrationObject`, `FT0DummyCalibrationObject`) - representing correction values

To conclude, data stored in calibration info object is required to fill TimeSlot container with data and that data stored in  the container is needed to generate calibration object.

###Implementation details
FITCalibration module is based on templates, so it can be used for FT0, FDD and FV0 calibration workflows. One has to define definition of the classes described above and also provide formula for calibration object generation.

###How to add new code, new calibration object?
- Define your calibration object class (can be combined more than one into one class)
- Define your timeslot container class (can be used for many calibration objects)
- Define algorithm / function for calibration object generation. Add template specialization with your classes in `FITCalibrationObjectProducer` `generateCalibrationObject` method
- Add template specialization for calibration object in `FitCalibrationApi`, define ccdb calib object path and also recipe how to store them in CCDB
- Create separated workflow or add your `DataProcessorSpec` to main calibration workflow

For example check `FT0CalibrationDummy-Workflow` which uses combined calibration object. It is also uses mechanism to retrieve (from CCDB) another calibration object needed for calibration. Before running this workflow it is required to run `makeDummyFt0CalibObjectInCCDB` macro to generate dummy object in the local CCDB instance.

###How to run these workflows?
- Dummy:
`root -l ~/alice/O2/Detectors/FIT/FT0/calibration/macros/makeDummyFT0CalibObjectInCCDB.C+`
  `o2-ft0-digits-reader-workflow | o2-calibration-ft0-tf-processor | o2-calibration-ft0-dummy-example | o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080 -b`
- Channel Time calibration:
`o2-ft0-digits-reader-workflow | o2-calibration-ft0-tf-processor | o2-calibration-ft0-channel-offset-calibration | o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080 -`
  
- General calibration workflow (for now, the same as Channel Time calibration)
`o2-ft0-digits-reader-workflow | o2-calibration-ft0-tf-processor | o2-calibration-ft0-calibration | o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080 -b`

You have to specify path to digits file in `o2-ft0-digits-reader-workflow`. If you don't need to store generated calib object you can delete last workflow part (`o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:8080 -b`)