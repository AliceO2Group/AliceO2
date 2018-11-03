cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2AliTPCCommonBase)
elseif(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE AliTPCCommonBase)
endif()

set(SRCS
    AliTPCCommon.cxx
)

set (HDRS
	AliTPCCommonDef.h
	AliTPCCommonDefGPU.h
    AliTPCCommonMath.h
	FlatObject.h
)

#Default cmake build script for AliRoot
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
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

    install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    Set(HEADERS ${HDRS})
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME AliTPCCommonBase_bucket)

    O2_GENERATE_LIBRARY()
endif()
