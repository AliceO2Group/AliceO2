cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(HLT_CUDA)
    add_subdirectory(GlobalTracker/cuda)
endif(HLT_CUDA)

if(HLT_OPENCL)
    add_subdirectory(GlobalTracker/opencl)
endif()

if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "O2")
set(MODULE O2TPCCAGPUTracking)
elseif(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "ALIROOT")
set(MODULE AliTPCCAGPUTracking)
endif()

include_directories(. SliceTracker Merger)

set(SRCS SliceTracker/AliHLTTPCCATrack.cxx 
    SliceTracker/AliHLTTPCCATrackParam.cxx
    SliceTracker/AliHLTTPCCATracklet.cxx
    SliceTracker/AliHLTTPCCAStartHitsFinder.cxx
    SliceTracker/AliHLTTPCCANeighboursCleaner.cxx
    SliceTracker/AliHLTTPCCAParam.cxx
    SliceTracker/AliHLTTPCCATracker.cxx
    SliceTracker/AliHLTTPCCATrackerFramework.cxx
    SliceTracker/AliHLTTPCCASliceData.cxx
    SliceTracker/AliHLTTPCCASliceOutput.cxx
    SliceTracker/AliHLTTPCCAStandaloneFramework.cxx
    SliceTracker/AliHLTTPCCATrackletConstructor.cxx
    SliceTracker/AliHLTTPCCANeighboursFinder.cxx
    SliceTracker/AliHLTTPCCAGrid.cxx
    SliceTracker/AliHLTTPCCATrackletSelector.cxx
    SliceTracker/AliHLTTPCCAHitArea.cxx
    SliceTracker/AliHLTTPCCAClusterData.cxx
    SliceTracker/AliHLTTPCCARow.cxx
    SliceTracker/AliHLTTPCCAGPUTracker.cxx
    Merger/AliHLTTPCGMMerger.cxx
    Merger/AliHLTTPCGMSliceTrack.cxx
    Merger/AliHLTTPCGMTrackParam.cxx
    Merger/AliHLTTPCGMPhysicalTrackModel.cxx
    Merger/AliHLTTPCGMPropagator.cxx
    Merger/AliHLTTPCGMPolynomialField.cxx
    Merger/AliHLTTPCGMPolynomialFieldCreator.cxx
)

#Some extra files / includes / settings needed for the build for AliRoot
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "ALIROOT")
    include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
    include_directories(${AliRoot_SOURCE_DIR}/HLT/BASE
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib
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
        SliceTracker/AliHLTTPCCATrackConvertor.cxx
        SliceTracker/AliHLTTPCCATrackerComponent.cxx
    )
    
    add_definitions(-DHLTCA_BUILD_ALIROOT_LIB)
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(HLTHeaders Standalone/include)
    add_definitions(-DHLTCA_STANDALONE)
endif()

#Some extra files / includes / settings needed for the build for O2
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "O2")
    set(SRCS ${SRCS}
        Interface/AliHLTTPCCAO2Interface.cxx
    )

    add_definitions(-DHLTCA_TPC_GEOMETRY_O2 -DHLTCA_BUILD_O2_LIB)
endif()

string(REPLACE ".cxx" ".h" HDRS "${SRCS}")
#Extra headers to install
set(HDRS ${HDRS}
    SliceTracker/AliHLTTPCCADef.h
    SliceTracker/AliHLTTPCCAMath.h
)

#Extra cpp files, whose headers we don't pass to CINT
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "O2")
    set(SRCS ${SRCS} Standalone/cmodules/timer.cpp)
endif()
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "ALIROOT")
    set (SRCS ${SRCS} ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx)
endif()

#Default cmake build script for AliRoot
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary("${MODULE}" "LinkDef_AliRoot.h" "${HDRS}" "${incdirs}")

    # Generate the ROOT map
    # Dependecies
    set(LIBDEPS STEERBase STEER ESD CDB HLTbase TPCbase TPCrec TPCcalib)
    generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/LinkDef_AliRoot.h")
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

    install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(${ALITPCCOMMON_BUILD_TYPE} STREQUAL "O2")
    Set(HEADERS ${HDRS})
    Set(LINKDEF LinkDef_O2.h)
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME TPCCAGPUTracking_bucket)

    O2_GENERATE_LIBRARY()

    set(TEST_SRCS
      ctest/testTPCCAGPUTracking.cxx
    )

    O2_GENERATE_TESTS(
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
      TEST_SRCS ${TEST_SRCS}
    )
endif()
