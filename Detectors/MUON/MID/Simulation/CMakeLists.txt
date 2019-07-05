set(MODULE_NAME "MIDSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ChamberEfficiencyResponse.cxx
  src/ChamberHV.cxx
  src/ChamberResponse.cxx
  src/ChamberResponseParams.cxx
  src/ClusterLabeler.cxx
  src/Detector.cxx
  src/Digitizer.cxx
  src/DigitsMerger.cxx
  src/DigitsPacker.cxx
  src/Geometry.cxx
  src/Hit.cxx
  src/Materials.cxx
  src/MCClusterLabel.cxx
  src/MCLabel.cxx
  src/PreClusterLabeler.cxx
  src/Stepper.cxx
  src/TrackLabeler.cxx
)

set(HEADERS
  include/${MODULE_NAME}/ChamberEfficiencyResponse.h
  include/${MODULE_NAME}/ChamberHV.h
  include/${MODULE_NAME}/ChamberResponse.h
  include/${MODULE_NAME}/ChamberResponseParams.h
  include/${MODULE_NAME}/ClusterLabeler.h
  include/${MODULE_NAME}/ColumnDataMC.h
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/Digitizer.h
  include/${MODULE_NAME}/DigitsMerger.h
  include/${MODULE_NAME}/DigitsPacker.h
  include/${MODULE_NAME}/Hit.h
  include/${MODULE_NAME}/MCClusterLabel.h
  include/${MODULE_NAME}/MCLabel.h
  include/${MODULE_NAME}/Stepper.h
  include/${MODULE_NAME}/TrackLabeler.h
)

set(LINKDEF src/MIDSimulationLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_simulation_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)
