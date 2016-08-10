set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/DataFormats/simulation
  ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/DataFormats/simulation/
  ${FAIRROOT_LIBRARY_DIR}
  ${ROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
  src/Stack.cxx
  src/MCTrack.cxx
)

Set(HEADERS
  include/Stack.h
  include/MCTrack.h
)
Set(LINKDEF src/SimulationDataLinkDef.h)
Set(LIBRARY_NAME O2SimulationDataFormat)
Set(DEPENDENCIES Base EG Physics Core)

GENERATE_LIBRARY()
