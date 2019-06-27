cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2GPUCommon)
elseif(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE GPUCommon)
endif()

set(SRCS
    GPUCommon.cxx
)

set (HDRS
    FlatObject.h
)

set (HDRS2
    GPUCommonAlgorithm.h
    GPUCommonDef.h
    GPUCommonDefAPI.h
    GPUCommonDefSettings.h
    GPUCommonFairLogger.h
    GPUCommonMath.h
    GPUCommonRtypes.h
    GPUCommonTransform3D.h
    GPUDef.h
    GPUDefConstantsAndSettings.h
    GPUDefGPUParameters.h
    GPUDefOpenCL12Templates.h
)

#Default cmake build script for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    # Add a library to the project using the specified source files
    add_library_tested(${MODULE} SHARED ${SRCS})

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
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    Set(HEADERS ${HDRS})
    Set(LINKDEF GPUCommonLinkDef.h)
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME GPUCommon_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} ${HDRS2} DESTINATION include/GPU)
endif()
