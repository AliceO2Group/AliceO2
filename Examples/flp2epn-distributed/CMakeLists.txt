configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}//test/testFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)

set(MODULE_NAME "FLP2EPNex_distributed")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/FLPSyncSampler.cxx
    src/FLPSender.cxx
    src/EPNReceiver.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/FLPSyncSampler.h
    include/${MODULE_NAME}/FLPSender.h
    include/${MODULE_NAME}/EPNReceiver.h)

if (DDS_FOUND)
  set(BUCKET_NAME flp2epndistrib_Bucket)
else ()
  set(BUCKET_NAME flp2epn_Bucket)
endif ()
set(LIBRARY_NAME FLP2EPNex_distributed)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)

Set(Exe_Names
    ${Exe_Names}
    flpSyncSampler
    flpSender
    epnReceiver
    )

if (DDS_FOUND)
  set(Exe_Names
      ${Exe_Names}
      flpSyncSampler_dds
      flpSender_dds
      epnReceiver_dds
      )
endif ()

set(Exe_Source
    run/runFLPSyncSampler.cxx
    run/runFLPSender.cxx
    run/runEPNReceiver.cxx
    )

if (DDS_FOUND)
  set(Exe_Source
      ${Exe_Source}
      runO2Prototype/runFLPSyncSampler.cxx
      runO2Prototype/runFLPSender.cxx
      runO2Prototype/runEPNReceiver.cxx
      )
endif ()

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
if (DDS_FOUND)
  set(BUCKET_NAME flp2epndistrib_Bucket)
else ()
  set(BUCKET_NAME flp2epn_Bucket)
endif ()
set(LIBRARY_NAME FLP2EPNex_distributed)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach (_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
EndForEach (_file RANGE 0 ${_length})

add_test(NAME run_flp2epn_distributed COMMAND ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)
set_tests_properties(run_flp2epn_distributed PROPERTIES TIMEOUT "30")
set_tests_properties(run_flp2epn_distributed PROPERTIES PASS_REGULAR_EXPRESSION "acknowledged after")
