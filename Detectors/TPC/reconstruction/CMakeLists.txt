option(HLT_CUDA   "Build HLT GPU tracker using CUDA"   OFF)
option(HLT_OPENCL "Build HLT GPU tracker using OpenCL" OFF)

if(DEFINED ALITPCCOMMON_DIR)
    set(ALITPCCOMMON_BUILD_TYPE "O2")
    add_subdirectory(${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking TPCCAGPUTracking)
    include_directories(${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/SliceTracker
                        ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Merger
                        ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Interface)
else()
    message(FATAL_ERROR "cmake variable ALITPCCOMMON_DIR is not defined")
endif()

set(MODULE_NAME "TPCReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/AdcClockMonitor.cxx
   src/BoxClusterer.cxx
   src/ClusterContainer.cxx
   src/Clusterer.cxx
   src/ClustererTask.cxx
   src/GBTFrame.cxx
   src/GBTFrameContainer.cxx
   src/HalfSAMPAData.cxx
   src/HwClusterer.cxx
   src/RawReader.cxx
   src/RawReaderEventSync.cxx
   src/SyncPatternMonitor.cxx
   src/TPCCATracking.cxx
   src/HardwareClusterDecoder.cxx
   src/DigitalCurrentClusterIntegrator.cxx
)

set(HEADERS
   include/${MODULE_NAME}/AdcClockMonitor.h
   include/${MODULE_NAME}/BoxClusterer.h
   include/${MODULE_NAME}/ClusterContainer.h
   include/${MODULE_NAME}/Clusterer.h
   include/${MODULE_NAME}/ClustererTask.h
   include/${MODULE_NAME}/GBTFrame.h
   include/${MODULE_NAME}/GBTFrameContainer.h
   include/${MODULE_NAME}/HalfSAMPAData.h
   include/${MODULE_NAME}/HwClusterer.h
   include/${MODULE_NAME}/RawReader.h
   include/${MODULE_NAME}/RawReaderEventSync.h
   include/${MODULE_NAME}/SyncPatternMonitor.h
   include/${MODULE_NAME}/TPCCATracking.h
   include/${MODULE_NAME}/HardwareClusterDecoder.h
   include/${MODULE_NAME}/DigitalCurrentClusterIntegrator.h
)
set(LINKDEF src/TPCReconstructionLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_reconstruction_bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
  tpc-read-GBTFrames
  tpc-read-RawData
)


Set(Exe_Source
  run/readGBTFrames.cxx
  run/readRawData.cxx
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
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
