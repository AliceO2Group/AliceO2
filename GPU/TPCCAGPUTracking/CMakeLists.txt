cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    add_definitions(-DHLTCA_BUILD_ALIROOT_LIB)
else()
    add_definitions(-DHLTCA_STANDALONE)
endif()

if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    add_definitions(-DHLTCA_BUILD_O2_LIB -DHLTCA_TPC_GEOMETRY_O2 -DHAVE_O2HEADERS)
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "AliTPCCommon: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
    add_definitions(-DHLTCA_HAVE_OPENMP)
endif()

if (ENABLE_CUDA OR ENABLE_OPENCL)
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
        message(WARNING "GPU Tracking disabled on MacOS")
    else()
        add_definitions(-DHLTCA_ENABLE_GPU_TRACKER)
        if(ENABLE_CUDA)
            add_subdirectory(GlobalTracker/cuda)
        endif(ENABLE_CUDA)
        if(ENABLE_OPENCL)
            add_subdirectory(GlobalTracker/opencl)
        endif(ENABLE_OPENCL)
    endif()
endif()

if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2TPCCAGPUTracking)
elseif(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE AliTPCCAGPUTracking)
endif()

include_directories(. SliceTracker Merger ../Common ../TPCFastTransformation Standalone/include)

#SRCs processed by CINT and added to HDRS
set(SRCS
    SliceTracker/AliHLTTPCCATrack.cxx
    SliceTracker/AliHLTTPCCABaseTrackParam.cxx
    SliceTracker/AliHLTTPCCATrackParam.cxx
    SliceTracker/AliHLTTPCCATracklet.cxx
    SliceTracker/AliHLTTPCCAStartHitsFinder.cxx
    SliceTracker/AliHLTTPCCANeighboursCleaner.cxx
    SliceTracker/AliHLTTPCCATracker.cxx
    SliceTracker/AliHLTTPCCASliceData.cxx
    SliceTracker/AliHLTTPCCASliceOutput.cxx
    SliceTracker/AliHLTTPCCATrackletConstructor.cxx
    SliceTracker/AliHLTTPCCANeighboursFinder.cxx
    SliceTracker/AliHLTTPCCAGrid.cxx
    SliceTracker/AliHLTTPCCATrackletSelector.cxx
    SliceTracker/AliHLTTPCCAHitArea.cxx
    SliceTracker/AliHLTTPCCAClusterData.cxx
    SliceTracker/AliHLTTPCCARow.cxx
    Merger/AliHLTTPCGMMerger.cxx
    Merger/AliHLTTPCGMSliceTrack.cxx
    Merger/AliHLTTPCGMTrackParam.cxx
    Merger/AliHLTTPCGMPhysicalTrackModel.cxx
    Merger/AliHLTTPCGMPropagator.cxx
    Merger/AliHLTTPCGMPolynomialField.cxx
    Merger/AliHLTTPCGMPolynomialFieldManager.cxx
    TRDTracking/AliHLTTRDTrack.cxx
    TRDTracking/AliHLTTRDTracker.cxx
    TRDTracking/AliHLTTRDTrackletWord.cxx
    GlobalTracker/AliGPUCAParam.cxx
)

set(SRCS_NO_CINT
    GlobalTracker/AliGPUReconstruction.cxx
    GlobalTracker/AliGPUReconstructionDeviceBase.cxx
    GlobalTracker/AliGPUReconstructionConvert.cxx
    GlobalTracker/AliGPUCASettings.cxx
)

#Extra headers to install
set(HDRS
    SliceTracker/AliHLTTPCCADef.h
    SliceTracker/AliHLTTPCCASettings.h
    SliceTracker/AliHLTTPCCAGPUConfig.h
    SliceTracker/AliHLTTPCCAMCInfo.h
    SliceTracker/AliHLTTPCCAHit.h
    SliceTracker/AliHLTTPCCAHitId.h
    SliceTracker/AliHLTTPCCASliceOutCluster.h
    SliceTracker/AliHLTTPCCASliceOutTrack.h
    SliceTracker/AliHLTTPCCATrackLinearisation.h
    Merger/AliHLTTPCGMBorderTrack.h
    Merger/AliHLTTPCGMCluster.h
    Merger/AliHLTTPCGMOfflineStatisticalErrors.h
    Merger/AliHLTTPCGMMergedTrack.h
    Merger/AliHLTTPCGMMergedTrackHit.h
    TRDTracking/AliHLTTRDDef.h
    TRDTracking/AliHLTTRDTrackPoint.h
    TRDTracking/AliHLTTRDTrack.h
    TRDTracking/AliHLTTRDTrackData.h
    TRDTracking/AliHLTTRDTrackPoint.h
    TRDTracking/AliHLTTRDTrackletLabels.h
    GlobalTracker/AliGPUCADataTypes.h
    GlobalTracker/AliGPUCAOutputControl.h
)
#Not processed by ROOT
set(HDRS2
    SliceTracker/AliHLTTPCCAProcess.h
    SliceTracker/AliHLTTPCCAGeometry.h
    SliceTracker/AliCAGPULogging.h
    SliceTracker/ClusterNativeAccessExt.h
    TRDTracking/AliHLTTRDGeometry.h
    TRDTracking/AliHLTTRDTrackerDebug.h
    TRDTracking/AliHLTTRDInterfaces.h
)

#Some extra files / includes / settings needed for the build for AliRoot
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
    include_directories(${AliRoot_SOURCE_DIR}/HLT/BASE
        ${AliRoot_SOURCE_DIR}/HLT/BASE/util
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/transform
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/comp
        ${AliRoot_SOURCE_DIR}/HLT/TRD
        ${AliRoot_SOURCE_DIR}/TRD/TRDbase
        ${AliRoot_SOURCE_DIR}/STEER/STEERBase
        ${AliRoot_SOURCE_DIR}/STEER/STEER
        ${AliRoot_SOURCE_DIR}/STEER/ESD
        ${AliRoot_SOURCE_DIR}/STEER/CDB
        ${AliRoot_SOURCE_DIR}/TPC/TPCbase
        ${AliRoot_SOURCE_DIR}/TPC/TPCcalib
        ${AliRoot_SOURCE_DIR}/TPC/TPCrec
    )

    set(SRCS ${SRCS}
        SliceTracker/AliHLTTPCCAGlobalMergerComponent.cxx
        SliceTracker/AliHLTTPCCATrackerComponent.cxx
        Merger/AliHLTTPCGMTracksToTPCSeeds.cxx
        DataCompression/AliHLTTPCClusterStatComponent.cxx
        TRDTracking/AliHLTTRDTrackerComponent.cxx
        TRDTracking/AliHLTTRDTrackletReaderComponent.cxx

        GlobalTracker/AliHLTGPUDumpComponent.cxx
    )
    
    set (SRCS_NO_CINT ${SRCS_NO_CINT}
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCDefinitions.cxx
        ${AliRoot_SOURCE_DIR}/HLT/TRD/AliHLTTRDDefinitions.cxx
    )
    
    set (HDRS2 ${HDRS2}
        SliceTracker/AliHLTTPCCADefinitions.h
    )
    
    ALICE_UseVc()
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(HLTHeaders Standalone/include)
endif()

#Some extra files / includes / settings needed for the build for O2
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    set(SRCS ${SRCS}
        Interface/AliHLTTPCCAO2Interface.cxx
    )
    set(SRCS_NO_CINT ${SRCS_NO_CINT}
        Standalone/cmodules/timer.cpp
    )
endif()

string(REPLACE ".cxx" ".h" HDRSSRC "${SRCS}")
set(HDRS ${HDRS} ${HDRSSRC})
string(REPLACE ".cxx" ".h" HDRSSRC "${SRCS_NO_CINT}")
set(HDRS2 ${HDRS2} ${HDRSSRC})
set(SRCS ${SRCS} ${SRCS_NO_CINT})

#Default cmake build script for AliRoot
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary_flat("${MODULE}" "TPCCAGPUTrackingLinkDef_AliRoot.h" "${HDRS}" "${incdirs}")

    # Generate the ROOT map
    # Dependecies
    set(LIBDEPS STEERBase STEER ESD CDB HLTbase TPCbase TPCrec TPCcalib TRDbase AliTPCFastTransformation)
    generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/TPCCAGPUTrackingLinkDef_AliRoot.h")
    # Don't pass Vc to root
    set(LIBDEPS ${LIBDEPS} Vc)

    # Add a library to the project using the specified source files
    add_library_tested(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
    target_link_libraries(${MODULE} ${LIBDEPS})

    # Additional compilation flags
    set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

    # System dependent: Modify the way the library is build
    if(${CMAKE_SYSTEM} MATCHES Darwin)
        set_target_properties(${MODULE} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif(${CMAKE_SYSTEM} MATCHES Darwin)

    # Installation
    install(TARGETS ${MODULE}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)

    install(FILES ${HDRS} ${HDRS2} DESTINATION include)
endif()

#Default cmake build script for O2
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    Set(HEADERS ${HDRS})
    Set(LINKDEF TPCCAGPUTrackingLinkDef_O2.h)
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME TPCCAGPUTracking_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} ${HDRS2} DESTINATION include/AliTPCCommon)

    set(TEST_SRCS
      ctest/testTPCCAGPUTracking.cxx
    )

    O2_GENERATE_TESTS(
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
      TEST_SRCS ${TEST_SRCS}
    )
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${MODULE} OpenMP::OpenMP_CXX)
    endif()
endif()
