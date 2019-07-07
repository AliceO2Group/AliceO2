o2_add_library(FLP2EPNex_distributed
               SOURCES src/FLPSyncSampler.cxx src/FLPSender.cxx
                       src/EPNReceiver.cxx
               PUBLIC_LINK_LIBRARIES O2::Device)

o2_add_executable(flp-sync-sampler
                  SOURCES run/runFLPSyncSampler.cxx
                  PUBLIC_LINK_LIBRARIES O2::FLP2EPNex_distributed
                  COMPONENT_NAME example)

o2_add_executable(flp-sender
                  SOURCES run/runFLPSender.cxx
                  PUBLIC_LINK_LIBRARIES O2::FLP2EPNex_distributed
                  COMPONENT_NAME example)

o2_add_executable(epn-receiver
                  SOURCES run/runEPNReceiver.cxx
                  PUBLIC_LINK_LIBRARIES O2::FLP2EPNex_distributed
                  COMPONENT_NAME example)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/startFLP2EPN-distributed.sh.in
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/o2-start-flp2epn-distributed.sh)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test/testFLP2EPN-distributed.sh.in
  ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/o2-test-flp2epn-distributed.sh
  )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-prototype.json
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-prototype.json)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-prototype-dds.json
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-prototype-dds.json)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-dds-topology.xml
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-dds-topology.xml
               @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/flp2epn-dds-hosts.cfg
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-dds-hosts.cfg
               COPYONLY)

# add_test_wrap(NAME run_flp2epn_distributed COMMAND
# ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-
# distributed.sh) set_tests_properties(run_flp2epn_distributed PROPERTIES
# TIMEOUT "30") set_tests_properties(run_flp2epn_distributed PROPERTIES
# PASS_REGULAR_EXPRESSION "acknowledged after")

install(
  FILES
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/o2-start-flp2epn-distributed.sh
    ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/o2-test-flp2epn-distributed.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-prototype.json
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-prototype-dds.json
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-dds-topology.xml
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn-dds-hosts.cfg
  DESTINATION ${CMAKE_INSTALL_DATADIR}/config)
