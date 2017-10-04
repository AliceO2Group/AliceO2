set(MODULE_NAME "TPCReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/AdcClockMonitor.cxx
   src/GBTFrame.cxx
   src/GBTFrameContainer.cxx
   src/HalfSAMPAData.cxx
   src/RawReader.cxx
   src/SyncPatternMonitor.cxx
   src/TrackTPC.cxx
   src/TPCCATracking.cxx
)

set(HEADERS
   include/${MODULE_NAME}/AdcClockMonitor.h
   include/${MODULE_NAME}/GBTFrame.h
   include/${MODULE_NAME}/GBTFrameContainer.h
   include/${MODULE_NAME}/HalfSAMPAData.h
   include/${MODULE_NAME}/RawReader.h
   include/${MODULE_NAME}/SyncPatternMonitor.h
   include/${MODULE_NAME}/TrackTPC.h
   include/${MODULE_NAME}/TPCCATracking.h
)
Set(LINKDEF src/TPCReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_reconstruction_bucket)

if(DEFINED O2_TPCCA_TRACKING_LIB_DIR)
  include_directories(${O2_TPCCA_TRACKING_LIB_DIR}/include)
  add_definitions(-DHAVE_O2_TPCCA_TRACKING_LIB -DHLTCA_STANDALONE -DHLTCA_TPC_GEOMETRY_O2)
  FIND_LIBRARY(CATRACKING_LIB libO2TPCCATracking.so ${O2_TPCCA_TRACKING_LIB_DIR}/lib)
  link_libraries(${CATRACKING_LIB})
  message("Building TPCCA tracking interface from standalone library in ${O2_TPCCA_TRACKING_LIB_DIR}")
else()
  message("Not building TPCCA tracking interface")
endif()

O2_GENERATE_LIBRARY()

Set(Exe_Names
  tpc-read-GBTFrames
  tpc-read-RawData
  testRawReader
)


Set(Exe_Source
  run/readGBTFrames.cxx
  run/readRawData.cxx
  run/testRawReader.cxx
)

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(BUCKET_NAME tpc_reconstruction_bucket)
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
  test/testTPCSyncPatternMonitor.cxx
  test/testTPCAdcClockMonitor.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
