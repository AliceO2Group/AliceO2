cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(DEFINITIONS ${DEFINITIONS} GPUCA_ALIROOT_LIB)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(DEFINITIONS ${DEFINITIONS} GPUCA_O2_LIB GPUCA_TPC_GEOMETRY_O2 HAVE_O2HEADERS)
    if (OPENGL_FOUND AND GLFW_FOUND AND GLEW_FOUND AND OPENGL_GLU_FOUND AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(GPUCA_EVENT_DISPLAY ON)
    endif()
    set(GPUCA_QA ON)
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "GPU: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
    set(DEFINITIONS ${DEFINITIONS} GPUCA_HAVE_OPENMP)
endif()

include_directories(. SliceTracker Merger Base Global TRDTracking ITS dEdx TPCConvert DataCompression ../Common ../TPCFastTransformation Standalone)

if (ENABLE_CUDA OR ENABLE_OPENCL OR ENABLE_HIP)
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
        message(WARNING "GPU Tracking disabled on MacOS")
    else()
        if(ENABLE_CUDA)
            add_subdirectory(Base/cuda)
        endif()
        if(ENABLE_OPENCL)
            add_subdirectory(Base/opencl)
        endif()
        if (ENABLE_HIP)
            add_subdirectory(Base/hip)
        endif()
    endif()
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2GPUTracking)
elseif(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE GPUTracking)
endif()

include_directories(Standalone/display Standalone/qa)

#SRCs processed by CINT and added to HDRS
set(SRCS
    SliceTracker/GPUTPCTrack.cxx
    SliceTracker/GPUTPCBaseTrackParam.cxx
    SliceTracker/GPUTPCTrackParam.cxx
    SliceTracker/GPUTPCTracklet.cxx
    SliceTracker/GPUTPCStartHitsFinder.cxx
    SliceTracker/GPUTPCStartHitsSorter.cxx
    SliceTracker/GPUTPCNeighboursCleaner.cxx
    SliceTracker/GPUTPCTracker.cxx
    SliceTracker/GPUTPCSliceData.cxx
    SliceTracker/GPUTPCSliceOutput.cxx
    SliceTracker/GPUTPCTrackletConstructor.cxx
    SliceTracker/GPUTPCNeighboursFinder.cxx
    SliceTracker/GPUTPCGrid.cxx
    SliceTracker/GPUTPCTrackletSelector.cxx
    SliceTracker/GPUTPCHitArea.cxx
    SliceTracker/GPUTPCClusterData.cxx
    SliceTracker/GPUTPCRow.cxx
    Merger/GPUTPCGMMerger.cxx
    Merger/GPUTPCGMSliceTrack.cxx
    Merger/GPUTPCGMTrackParam.cxx
    Merger/GPUTPCGMPropagator.cxx
    Merger/GPUTPCGMPhysicalTrackModel.cxx
    Merger/GPUTPCGMPolynomialField.cxx
    Merger/GPUTPCGMPolynomialFieldManager.cxx
    Merger/GPUTPCGMMergerGPU.cxx
    TRDTracking/GPUTRDTrack.cxx
    TRDTracking/GPUTRDTracker.cxx
    TRDTracking/GPUTRDTrackletWord.cxx
    TRDTracking/GPUTRDTrackerGPU.cxx
    Base/GPUParam.cxx
)

set(SRCS_NO_CINT
    Base/GPUDataTypes.cxx
    Base/GPUReconstruction.cxx
    Base/GPUReconstructionCPU.cxx
    Base/GPUSettings.cxx
    Base/GPUProcessor.cxx
    Base/GPUMemoryResource.cxx
    Base/GPUGeneralKernels.cxx
    Base/GPUReconstructionDeviceBase.cxx
    Base/GPUReconstructionConvert.cxx
    Global/GPUChain.cxx
    Global/GPUChainTracking.cxx
    Standalone/utils/timer.cpp
)

set(SRCS_NO_H
    SliceTracker/GPUTPCTrackerDump.cxx
    Global/GPUChainTrackingDebugAndProfiling.cxx
)

#Extra headers to install
set(HDRS
    SliceTracker/GPUTPCMCInfo.h
    SliceTracker/GPUTPCHit.h
    SliceTracker/GPUTPCHitId.h
    SliceTracker/GPUTPCSliceOutCluster.h
    SliceTracker/GPUTPCSliceOutTrack.h
    SliceTracker/GPUTPCTrackLinearisation.h
    Merger/GPUTPCGMBorderTrack.h
    Merger/GPUTPCGMOfflineStatisticalErrors.h
    Merger/GPUTPCGMMergedTrack.h
    Merger/GPUTPCGMMergedTrackHit.h
    TRDTracking/GPUTRDDef.h
    TRDTracking/GPUTRDTrackPoint.h
    TRDTracking/GPUTRDTrack.h
    TRDTracking/GPUTRDTrackData.h
    TRDTracking/GPUTRDTrackPoint.h
    TRDTracking/GPUTRDTrackletLabels.h
    Base/GPUReconstructionIncludes.h
)
#Not processed by ROOT
set(HDRS2
    SliceTracker/GPUTPCDef.h
    SliceTracker/GPUTPCGeometry.h
    SliceTracker/GPULogging.h
    SliceTracker/ClusterNativeAccessExt.h
    TRDTracking/GPUTRDGeometry.h
    TRDTracking/GPUTRDTrackerDebug.h
    TRDTracking/GPUTRDInterfaces.h
    Standalone/display/GPUDisplayConfig.h
    Standalone/qa/GPUQAConfig.h
    Base/GPUConstantMem.h
    Base/GPUReconstructionKernels.h
    Base/GPUReconstructionIncludesITS.h
    Base/GPUReconstructionHelpers.h
    Base/GPUOutputControl.h
)

#Some extra files / includes / settings needed for the build for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
    include_directories(${CMAKE_SOURCE_DIR}/HLT/BASE
        ${CMAKE_SOURCE_DIR}/HLT/BASE/util
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib/transform
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib/comp
        ${CMAKE_SOURCE_DIR}/HLT/TRD
        ${CMAKE_SOURCE_DIR}/TRD/TRDbase
        ${CMAKE_SOURCE_DIR}/STEER/STEERBase
        ${CMAKE_SOURCE_DIR}/STEER/STEER
        ${CMAKE_SOURCE_DIR}/STEER/ESD
        ${CMAKE_SOURCE_DIR}/STEER/CDB
        ${CMAKE_SOURCE_DIR}/TPC/TPCbase
        ${CMAKE_SOURCE_DIR}/TPC/TPCcalib
        ${CMAKE_SOURCE_DIR}/TPC/TPCrec
    )

    set(SRCS ${SRCS}
        Merger/GPUTPCGlobalMergerComponent.cxx
        SliceTracker/GPUTPCTrackerComponent.cxx
        Merger/GPUTPCGMTracksToTPCSeeds.cxx
        DataCompression/AliHLTTPCClusterStatComponent.cxx
        TRDTracking/GPUTRDTrackerComponent.cxx
        TRDTracking/GPUTRDTrackletReaderComponent.cxx

        Global/AliHLTGPUDumpComponent.cxx
    )
    
    set (SRCS_NO_CINT ${SRCS_NO_CINT}
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx
        ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCDefinitions.cxx
        ${CMAKE_SOURCE_DIR}/HLT/TRD/AliHLTTRDDefinitions.cxx
    )
    
    set (HDRS2 ${HDRS2}
        SliceTracker/GPUTPCDefinitions.h
    )
    
    ALICE_UseVc()
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(HLTHeaders)
endif()

#Some extra files / includes / settings needed for the build for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(SRCS ${SRCS}
        Interface/GPUO2Interface.cxx
    )
    set(SRCS_NO_CINT ${SRCS_NO_CINT}
        Standalone/display/GPUDisplayBackend.cpp
        Global/GPUChainITS.cxx
        ITS/GPUITSFitter.cxx
        ITS/GPUITSFitterKernels.cxx
        dEdx/GPUdEdx.cxx
        TPCConvert/GPUTPCConvert.cxx
        TPCConvert/GPUTPCConvertKernel.cxx
        DataCompression/GPUTPCCompression.cxx
        DataCompression/GPUTPCCompressionTrackModel.cxx
        DataCompression/GPUTPCCompressionKernels.cxx
        DataCompression/TPCClusterDecompressor.cxx
        DataCompression/GPUTPCClusterStatistics.cxx
    )
    set (HDRS2 ${HDRS2}
        Interface/GPUO2InterfaceConfiguration.h
        ITS/GPUITSTrack.h
        dEdx/GPUdEdxInfo.h
    )
else() #Need at least the header installed, as it is part of constant memory
    set (HDRS2 ${HDRS2}
        ITS/GPUITSFitter.h
        ITS/GPUITSFitterKernels.h
        ITS/GPUITSTrack.h
        TPCConvert/GPUTPCConvert.h
        TPCConvert/GPUTPCConvertKernel.h
        DataCompression/GPUTPCCompression.h
        DataCompression/GPUTPCCompressionTrackModel.h
        DataCompression/GPUTPCCompressionKernels.h
        DataCompression/TPCClusterDecompressor.h
        DataCompression/GPUTPCClusterStatistics.h
    )
endif()

if (GPUCA_EVENT_DISPLAY)
    message(STATUS "Building GPU Event Display")
    set(SRCS_NO_CINT ${SRCS_NO_CINT}
        Standalone/utils/qsem.cpp
        Standalone/display/GPUDisplay.cpp
        Standalone/display/GPUDisplayBackendGlfw.cpp
    )
    set(SRCS_NO_H ${SRCS_NO_H}
        Standalone/display/GPUDisplayQuaternion.cpp
        Standalone/display/GPUDisplayInterpolation.cpp
        Standalone/display/GPUDisplayKeys.cpp
    )
    set(DEFINITIONS ${DEFINITIONS} BUILD_EVENT_DISPLAY)
else()
    set(HDRS2 ${HDRS2} Standalone/display/GPUDisplay.h)
endif()

if (GPUCA_QA)
    message(STATUS "Building GPU QA")
    set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/qa/GPUQA.cpp)
    set(DEFINITIONS ${DEFINITIONS} BUILD_QA)
else()
    set(HDRS2 ${HDRS2} Standalone/qa/GPUQA.h)
endif()

if (GPUCA_EVENT_DISPLAY OR GPUCA_QA)
    set(HDRS2 ${HDRS2} Standalone/qconfigoptions.h)
endif()

string(REPLACE ".cxx" ".h" HDRSSRCA "${SRCS}")
string(REPLACE ".cpp" ".h" HDRSSRC "${HDRSSRCA}")
set(HDRS ${HDRS} ${HDRSSRC})
string(REPLACE ".cxx" ".h" HDRSSRCA "${SRCS_NO_CINT}")
string(REPLACE ".cpp" ".h" HDRSSRC "${HDRSSRCA}")
set(HDRS2 ${HDRS2} ${HDRSSRC})
set(SRCS ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})
foreach (def ${DEFINITIONS})
    add_definitions(-D${def})
endforeach()

#Default cmake build script for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary_flat("${MODULE}" "GPUTrackingLinkDef_AliRoot.h" "${HDRS}" "${incdirs}")

    # Generate the ROOT map
    # Dependecies
    set(LIBDEPS STEERBase STEER ESD CDB HLTbase TPCbase TPCrec TPCcalib TRDbase AliTPCFastTransformation)
    generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/GPUTrackingLinkDef_AliRoot.h")
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
    install(DIRECTORY Standalone/utils DESTINATION include FILES_MATCHING PATTERN *.h)
endif()

#Default cmake build script for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    Set(HEADERS ${HDRS})
    Set(LINKDEF GPUTrackingLinkDef_O2.h)
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME GPUTracking_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} ${HDRS2} DESTINATION include/GPU)
    install(DIRECTORY Standalone/utils DESTINATION include/GPU FILES_MATCHING PATTERN *.h)

    set(TEST_SRCS
      ctest/testGPUTracking.cxx
    )

    O2_GENERATE_TESTS(
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
      TEST_SRCS ${TEST_SRCS}
    )
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${MODULE} OpenMP::OpenMP_CXX)
    endif()
    if (GPUCA_EVENT_DISPLAY)
        target_link_libraries(${MODULE} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES} OpenGL::GL OpenGL::GLU)
    endif()
endif()
