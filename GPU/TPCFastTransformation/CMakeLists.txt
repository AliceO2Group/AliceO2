cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
set(MODULE O2TPCFastTransformation)
elseif(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
set(MODULE AliTPCFastTransformation)
add_definitions(-DGPUCA_ALIROOT_LIB)
endif()

include_directories(.)
include_directories(../Common)

set(SRCS
    IrregularSpline1D.cxx
    IrregularSpline2D3D.cxx
    SemiregularSpline2D3D.cxx
    IrregularSpline2D3DCalibrator.cxx
    TPCDistortionIRS.cxx
    TPCFastTransform.cxx
)

set (HDRS
    ../Common/GPUCommonDef.h
)

#Some extra files / includes / settings needed for the build for AliRoot
if(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
    set (SRCS ${SRCS}
        TPCFastTransformManager.cxx
        TPCFastTransformQA.cxx
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx
    )
    set (HDRS
        ${HDRS}
        TPCFastTransformManager.h
        TPCFastTransformQA.h
    )

    # Enable Vc
    ALICE_UseVc()

    include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
    include_directories(${AliRoot_SOURCE_DIR}/HLT/BASE
        ${AliRoot_SOURCE_DIR}/HLT/TPCLib
        ${AliRoot_SOURCE_DIR}/TPC/TPCbase
        ${AliRoot_SOURCE_DIR}/STEER/STEERBase
    )

endif()

if (NOT ROOT_VERSION_MAJOR EQUAL 5)
    set (HDRS ${HDRS}
        ../Common/FlatObject.h
        IrregularSpline1D.h
        IrregularSpline2D3D.h
	SemiregularSpline2D3D.h
 	IrregularSpline2D3DCalibrator.h
        RegularSpline1D.h
        TPCFastTransform.h
        TPCDistortionIRS.h
    )
endif()

#Default cmake build script for AliRoot
if(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary("${MODULE}" "TPCFastTransformationLinkDef_AliRoot.h" "${HDRS}" "${incdirs}")

    # Generate the ROOT map
    # Dependecies
    set(LIBDEPS STEERBase HLTbase TPCbase)
    generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/TPCFastTransformationLinkDef_AliRoot.h")
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
if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
    string(REPLACE ".cxx" ".h" HDRS "${SRCS}")
    set (HDRS
        ${HDRS}
    	../Common/GPUCommonDef.h
	RegularSpline1D.h
    )

    Set(HEADERS ${HDRS})
    Set(LINKDEF TPCFastTransformationLinkDef_O2.h)
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME TPCFastTransformation_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} DESTINATION include/GPU)

#    set(TEST_SRCS
#      ctest/testGPUTracking.cxx
#    )

#    O2_GENERATE_TESTS(
#      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
#      BUCKET_NAME ${BUCKET_NAME}
#      TEST_SRCS ${TEST_SRCS}
#    )
endif()
