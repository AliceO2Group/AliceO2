set(MODULE_NAME "TPCSimulation")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Baseline.cxx
   src/BoxCluster.cxx
   src/BoxClusterer.cxx
   src/Cluster.cxx
   src/ClusterContainer.cxx
   src/ClustererTask.cxx
   src/CommonMode.cxx
   src/Detector.cxx
   src/Digit.cxx
   src/DigitADC.cxx
   src/DigitContainer.cxx
   src/DigitCRU.cxx
   src/Digitizer.cxx
   src/DigitizerTask.cxx
   src/DigitPad.cxx
   src/DigitRow.cxx
   src/DigitTime.cxx
   src/GEMAmplification.cxx
   src/PadResponse.cxx
   src/Point.cxx
)

set(HEADERS
   include/${MODULE_NAME}/Baseline.h
   include/${MODULE_NAME}/BoxCluster.h
   include/${MODULE_NAME}/BoxClusterer.h
   include/${MODULE_NAME}/Cluster.h
   include/${MODULE_NAME}/ClusterContainer.h
   include/${MODULE_NAME}/ClustererTask.h
   include/${MODULE_NAME}/CommonMode.h
   include/${MODULE_NAME}/Detector.h
   include/${MODULE_NAME}/Digit.h
   include/${MODULE_NAME}/DigitADC.h
   include/${MODULE_NAME}/DigitContainer.h
   include/${MODULE_NAME}/DigitCRU.h
   include/${MODULE_NAME}/Digitizer.h
   include/${MODULE_NAME}/DigitizerTask.h
   include/${MODULE_NAME}/DigitPad.h
   include/${MODULE_NAME}/DigitRow.h
   include/${MODULE_NAME}/DigitTime.h
   include/${MODULE_NAME}/GEMAmplification.h
   include/${MODULE_NAME}/PadResponse.h
   include/${MODULE_NAME}/Point.h
)
Set(LINKDEF src/TPCSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_simulation_bucket)

O2_GENERATE_LIBRARY()
