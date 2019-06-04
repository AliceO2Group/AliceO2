set(MODULE_NAME "TPCReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/AdcClockMonitor.cxx
   src/ClustererTask.cxx
   src/GBTFrame.cxx
   src/GBTFrameContainer.cxx
   src/HalfSAMPAData.cxx
   src/HwClusterer.cxx
   src/RawReader.cxx
   src/RawReaderCRU.cxx
   src/RawReaderEventSync.cxx
   src/SyncPatternMonitor.cxx
   src/TPCCATracking.cxx
   src/HardwareClusterDecoder.cxx
   src/DigitalCurrentClusterIntegrator.cxx
   src/TPCFastTransformHelperO2.cxx
)

set(HEADERS
   include/${MODULE_NAME}/AdcClockMonitor.h
   include/${MODULE_NAME}/Clusterer.h
   include/${MODULE_NAME}/ClustererTask.h
   include/${MODULE_NAME}/GBTFrame.h
   include/${MODULE_NAME}/GBTFrameContainer.h
   include/${MODULE_NAME}/HalfSAMPAData.h
   include/${MODULE_NAME}/HwClusterer.h
   include/${MODULE_NAME}/RawReader.h
   include/${MODULE_NAME}/RawReaderCRU.h
   include/${MODULE_NAME}/RawReaderEventSync.h
   include/${MODULE_NAME}/SyncPatternMonitor.h
   include/${MODULE_NAME}/TPCCATracking.h
   include/${MODULE_NAME}/HardwareClusterDecoder.h
   include/${MODULE_NAME}/DigitalCurrentClusterIntegrator.h
   include/${MODULE_NAME}/TPCFastTransformHelperO2.h
)
set(LINKDEF src/TPCReconstructionLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_reconstruction_bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
  o2-tpc-read-gbtframes
  o2-tpc-read-rawdata
  o2-tpc-rawreader-cru
)


Set(Exe_Source
  run/readGBTFrames.cxx
  run/readRawData.cxx
  run/rawReaderCRU.cxx
)

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(BUCKET_NAME tpc_reconstruction_bucket)
set(LIBRARY_NAME ${MODULE_NAME})

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  # FIXME: propably a copy paste remnant, remove?
#  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

set(TEST_SRCS
  test/testTPCSyncPatternMonitor.cxx
  test/testTPCAdcClockMonitor.cxx
  test/testTPCCATracking.cxx
  test/testTPCHwClusterer.cxx
  test/testTPCFastTransform.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
