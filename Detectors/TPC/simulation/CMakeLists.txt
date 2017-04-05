set(MODULE_NAME "TPCSimulation")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Baseline.cxx
   src/BoxCluster.cxx
   src/BoxClusterer.cxx
   src/Cluster.cxx
   src/Clusterer.cxx
   src/ClusterContainer.cxx
   src/ClustererTask.cxx
   src/CommonMode.cxx
   src/Detector.cxx
   src/DigitMC.cxx
   src/DigitContainer.cxx
   src/DigitCRU.cxx
   src/Digitizer.cxx
   src/DigitizerTask.cxx
   src/DigitPad.cxx
   src/DigitRow.cxx
   src/DigitTime.cxx
   src/ElectronTransport.cxx
   src/GEMAmplification.cxx
   src/HwCluster.cxx
   src/HwClusterer.cxx
   src/HwClusterFinder.cxx
   src/HwFixedPoint.cxx
   src/PadResponse.cxx
   src/Point.cxx
   src/SAMPAProcessing.cxx
)

set(HEADERS
   include/${MODULE_NAME}/Baseline.h
   include/${MODULE_NAME}/BoxCluster.h
   include/${MODULE_NAME}/BoxClusterer.h
   include/${MODULE_NAME}/Cluster.h
   include/${MODULE_NAME}/Clusterer.h
   include/${MODULE_NAME}/ClusterContainer.h
   include/${MODULE_NAME}/ClustererTask.h
   include/${MODULE_NAME}/CommonMode.h
   include/${MODULE_NAME}/Detector.h
   include/${MODULE_NAME}/DigitMC.h
   include/${MODULE_NAME}/DigitContainer.h
   include/${MODULE_NAME}/DigitCRU.h
   include/${MODULE_NAME}/Digitizer.h
   include/${MODULE_NAME}/DigitizerTask.h
   include/${MODULE_NAME}/DigitPad.h
   include/${MODULE_NAME}/DigitRow.h
   include/${MODULE_NAME}/DigitTime.h
   include/${MODULE_NAME}/ElectronTransport.h
   include/${MODULE_NAME}/GEMAmplification.h
   include/${MODULE_NAME}/HwCluster.h
   include/${MODULE_NAME}/HwClusterer.h
   include/${MODULE_NAME}/HwClusterFinder.h
   include/${MODULE_NAME}/HwFixedPoint.h
   include/${MODULE_NAME}/PadResponse.h
   include/${MODULE_NAME}/Point.h
   include/${MODULE_NAME}/SAMPAProcessing.h
)
Set(LINKDEF src/TPCSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_simulation_bucket)

O2_GENERATE_LIBRARY()

install(
  DIRECTORY files
  DESTINATION share/Detectors/TPC/
)

Set(Exe_Names
  tpc-run-sim
)

Set(Exe_Source
  run/runSim.cxx
)

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(BUCKET_NAME tpc_simulation_bucket)
set(LIBRARY_NAME ${MODULE_NAME})

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
#  Set(EXE_NAME ${_name})
#  Set(SRCS ${_src})
#  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

set(TEST_SRCS
   test/testTPCDigitContainer.cxx
   test/testTPCElectronTransport.cxx
   test/testTPCGEMAmplification.cxx
   test/testTPCSAMPAProcessing.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
