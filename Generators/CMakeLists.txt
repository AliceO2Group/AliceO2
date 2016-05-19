# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

if(PYTHIA8_INCLUDE_DIR)



#    set(INCLUDE_DIRECTORIES
#      ${CMAKE_SOURCE_DIR}/Generators/include
#      ${CMAKE_SOURCE_DIR}/Generators/
#    )
#
#    set(SYSTEM_INCLUDE_DIRECTORIES
#      ${ROOT_INCLUDE_DIR}
#      ${FAIRROOT_INCLUDE_DIR}
#      ${PYTHIA8_INCLUDE_DIR}
#      ${SIMPATH}/include
#      ${BASE_INCLUDE_DIRECTORIES}
#    )
#
#    include_directories( ${INCLUDE_DIRECTORIES})
#    include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})
#
#    set(LINK_DIRECTORIES
#      ${ROOT_LIBRARY_DIR}
#      ${FAIRROOT_LIBRARY_DIR}
#      ${Pythia6_LIBRARY_DIR}
#      ${PYTHIA8_LIBRARY_DIR}
#      ${SIMPATH}/lib
#    )
#
#    link_directories( ${LINK_DIRECTORIES})
#
#    set(SRCS
#      src/Pythia6Generator.cxx
#      src/Pythia8Generator.cxx
#    )
#
#    set( HEADERS
#      include/Pythia6Generator.h
#      include/Pythia8Generator.h
#    )
#
#    set(LINKDEF src/EventGeneratorLinkDef.h)
#    set(LIBRARY_NAME EvtGenerator)
#
#    set(DEPENDENCIES Base O2SimulationDataFormat pythia8 Pythia6)
#
#    GENERATE_LIBRARY()



else(PYTHIA8_INCLUDE_DIR)
  message(STATUS "module 'Generators' requires Pythia8 ... deactivated")
endif(PYTHIA8_INCLUDE_DIR)
