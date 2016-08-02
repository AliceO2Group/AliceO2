set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/base/include
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/simulation/include
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/simulation/

  )

set(SYSTEM_INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
  ${ROOT_INCLUDE_DIR}
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
  src/GeometryTGeo.cxx
)
set(HEADERS
  include/Detector.h
  include/GeometryTGeo.h
)

Set(LINKDEF src/mftSimulationLinkDef.h)
Set(LIBRARY_NAME mftSimulation)
Set(DEPENDENCIES
    mftBase
    AliceO2Base
    Base ParBase
)

GENERATE_LIBRARY()
