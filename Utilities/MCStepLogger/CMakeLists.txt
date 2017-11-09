# @author Sandro Wenzel
# @brief  cmake setup for module Utilities/MCStepLogger

set(MODULE_NAME "MCStepLogger")
set(MODULE_BUCKET_NAME mcsteplogger_bucket)
O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/MCStepInterceptor.cxx
    src/MCStepLoggerImpl.cxx
    src/StepInfo.cxx
   )

set(HEADERS
   src/StepInfo.h
  )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})
set(LINKDEF src/MCStepLoggerLinkDef.h)

O2_GENERATE_LIBRARY()

# check correct functioning of the logger
if (HAVESIMULATION)
  add_test(NAME mcloggertest COMMAND ${CMAKE_BINARY_DIR}/bin/runTPC -n 1 -e TGeant3)
  # tests if the logger was active
  set_tests_properties(mcloggertest PROPERTIES PASS_REGULAR_EXPRESSION "VolName.*COUNT")
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(PRELOAD "DYLD_INSERT_LIBRARIES")
  else()
    set(PRELOAD "LD_PRELOAD")
  endif()
    set_tests_properties(mcloggertest PROPERTIES ENVIRONMENT ${PRELOAD}=${CMAKE_BINARY_DIR}/lib/libMCStepLogger${CMAKE_SHARED_LIBRARY_SUFFIX})
    set_property(TEST mcloggertest APPEND PROPERTY ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR})
endif()
