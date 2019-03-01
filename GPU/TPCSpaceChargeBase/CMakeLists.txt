cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
set(MODULE O2TPCSpaceChargeBase)
elseif(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
set(MODULE AliTPCSpaceChargeBase)
endif()

include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
include_directories(.)

set(SRCS
  AliTPC3DCylindricalInterpolator.cxx
  AliTPC3DCylindricalInterpolatorIrregular.cxx
  AliTPCLookUpTable3DInterpolatorD.cxx
  AliTPCLookUpTable3DInterpolatorIrregularD.cxx
  AliTPCPoissonSolver.cxx
  AliTPCSpaceCharge3DCalc.cxx
)
string(REPLACE ".cxx" ".h" HDRS "${SRCS}")

#Default cmake build script for AliRoot
if(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
  # Generate the dictionary
  # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("${MODULE}" "TPCSpaceChargeBaseLinkDef.h" "${HDRS}" "${incdirs}")

  set(ROOT_DEPENDENCIES Core Hist MathCore Matrix Physics)

  # Generate the ROOT map
  # Dependecies
  set(LIBDEPS ${ROOT_DEPENDENCIES})
  generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/TPCSpaceChargeBaseLinkDef.h")

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
    LIBRARY DESTINATION lib
  )

  install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
  Set(HEADERS ${HDRS})
  Set(LINKDEF TPCSpaceChargeBaseLinkDef.h)
  Set(LIBRARY_NAME ${MODULE})
  set(BUCKET_NAME TPCSpaceChargeBase_bucket)

  O2_GENERATE_LIBRARY()
  install(FILES ${HDRS} DESTINATION include/GPU)

  set(TEST_SRCS
    ctest/testTPCSpaceChargeBase.cxx
  )

  O2_GENERATE_TESTS(
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
    TEST_SRCS ${TEST_SRCS}
  )
endif()
