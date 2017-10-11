# Cleanup when API stabilizes
add_definitions(-DSHM_MULTIPART=1)
add_definitions(-DSTF_SERIALIZATION=2)

add_subdirectory (Common)
add_subdirectory (ReadoutEmulator)
add_subdirectory (SubTimeFrameBuilder)
add_subdirectory (SubTimeFrameTransporter)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/startReadoutFLPChain.sh.in ${CMAKE_BINARY_DIR}/bin/startReadoutFLPChain.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/readout-flp-chain.json ${CMAKE_BINARY_DIR}/bin/config/readout-flp-chain.json)
