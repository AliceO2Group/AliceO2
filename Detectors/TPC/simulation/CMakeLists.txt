
set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/Detectors/tpc
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/include
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
)
include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Detector.cxx
   src/Point.cxx

)

set(HEADERS
   include/Detector.h
   include/Point.h
)
Set(LINKDEF src/tpcSimulationLinkDef.h)
Set(LIBRARY_NAME tpcSimulation)
Set(DEPENDENCIES
    tpcBase
)
GENERATE_LIBRARY()
