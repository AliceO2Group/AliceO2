configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test/testFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-prototype.json ${CMAKE_BINARY_DIR}/bin/config/flp2epn-prototype.json)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-prototype-dds.json ${CMAKE_BINARY_DIR}/bin/config/flp2epn-prototype-dds.json)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-dds-topology.xml ${CMAKE_BINARY_DIR}/bin/config/flp2epn-dds-topology.xml @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-dds-hosts.cfg ${CMAKE_BINARY_DIR}/bin/config/flp2epn-dds-hosts.cfg COPYONLY)

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
  set(BUCKET_NAME flp2epndistrib_bucket)
else ()
  set(BUCKET_NAME flp2epn_bucket)
endif ()
set(LIBRARY_NAME ${MODULE_NAME})

O2_GENERATE_LIBRARY()

Set(Exe_Names
    ${Exe_Names}
    flpSyncSampler
    flpSender
    epnReceiver
    )

set(Exe_Source
    run/runFLPSyncSampler.cxx
    run/runFLPSender.cxx
    run/runEPNReceiver.cxx
    )

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

#add_test(NAME run_flp2epn_distributed COMMAND ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)
#set_tests_properties(run_flp2epn_distributed PROPERTIES TIMEOUT "30")
#set_tests_properties(run_flp2epn_distributed PROPERTIES PASS_REGULAR_EXPRESSION "acknowledged after")

install(FILES ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh
              ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(FILES ${CMAKE_BINARY_DIR}/bin/config/flp2epn-prototype.json
              ${CMAKE_BINARY_DIR}/bin/config/flp2epn-prototype-dds.json
              ${CMAKE_BINARY_DIR}/bin/config/flp2epn-dds-topology.xml
              ${CMAKE_BINARY_DIR}/bin/config/flp2epn-dds-hosts.cfg
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/config)
