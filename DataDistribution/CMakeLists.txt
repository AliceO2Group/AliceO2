add_subdirectory(Common)
add_subdirectory(ReadoutEmulator)
add_subdirectory(SubTimeFrameBuilder)
add_subdirectory(SubTimeFrameSender)
add_subdirectory(TimeFrameBuilder)

add_subdirectory(test)

# (FLP+EPN) Chain configuration and run script for Readout-Emulation mode (no CRU)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/start_Emulator-3FLP-3EPN.sh.in
          ${CMAKE_BINARY_DIR}/bin/start_Emulator-3FLP-3EPN.sh
)

install(
  PROGRAMS ${CMAKE_BINARY_DIR}/bin/start_Emulator-3FLP-3EPN.sh
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

## conf
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/readout_cfg/readout_emu.cfg
   ${CMAKE_BINARY_DIR}/bin/config/readout_cfg/readout_emu.cfg
)

install(FILES ${CMAKE_BINARY_DIR}/bin/config/readout_cfg/readout_emu.cfg
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/config
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/readout-emu-flp-epn-chain.json
   ${CMAKE_BINARY_DIR}/bin/config/readout-emu-flp-epn-chain.json
)

install(FILES ${CMAKE_BINARY_DIR}/bin/config/readout-emu-flp-epn-chain.json
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/config
)

# DDS topologies
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run/flp-epn-dds-topology.xml
   ${CMAKE_BINARY_DIR}/bin/config/flp-epn-dds-topology.xml COPYONLY
)

install(FILES ${CMAKE_BINARY_DIR}/bin/config/flp-epn-dds-topology.xml
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/config
)

