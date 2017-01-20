set(MODULE_NAME "TPCSimulation")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Detector.cxx
   src/Point.cxx
   src/DigitizerTask.cxx
   src/Digitizer.cxx
   src/DigitContainer.cxx
   src/DigitCRU.cxx
   src/DigitRow.cxx
   src/DigitPad.cxx
   src/DigitTime.cxx
   src/DigitADC.cxx
   src/Digit.cxx
   src/Cluster.cxx
   src/BoxCluster.cxx
   src/ClusterContainer.cxx
   src/BoxClusterer.cxx
   src/ClustererTask.cxx
   src/PadResponse.cxx
   src/CommonMode.cxx
   src/Baseline.cxx
   src/GEMAmplification.cxx
)

set(HEADERS
   include/${MODULE_NAME}/Detector.h
   include/${MODULE_NAME}/Point.h
   include/${MODULE_NAME}/DigitizerTask.h
   include/${MODULE_NAME}/Digitizer.h
   include/${MODULE_NAME}/DigitContainer.h
   include/${MODULE_NAME}/DigitCRU.h
   include/${MODULE_NAME}/DigitRow.h
   include/${MODULE_NAME}/DigitPad.h
   include/${MODULE_NAME}/DigitTime.h
   include/${MODULE_NAME}/DigitADC.h
   include/${MODULE_NAME}/Digit.h
   include/${MODULE_NAME}/Cluster.h
   include/${MODULE_NAME}/BoxCluster.h
   include/${MODULE_NAME}/ClusterContainer.h
   include/${MODULE_NAME}/BoxClusterer.h
   include/${MODULE_NAME}/ClustererTask.h
   include/${MODULE_NAME}/PadResponse.h
   include/${MODULE_NAME}/CommonMode.h
   include/${MODULE_NAME}/Baseline.h
   include/${MODULE_NAME}/GEMAmplification.h
)
Set(LINKDEF src/TPCSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_simulation_bucket)

O2_GENERATE_LIBRARY()
