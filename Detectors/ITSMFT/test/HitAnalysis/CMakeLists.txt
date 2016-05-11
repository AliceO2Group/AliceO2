set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/simulation/include
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/test/HitAnalysis/include
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/test/HitAnalysis/

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
  src/HitAnalysis.cxx
)
set(HEADERS
  include/HitAnalysis.h
)
Set(LINKDEF src/testitsLinkDef.h)
Set(LIBRARY_NAME testits)
Set(DEPENDENCIES
    AliceO2Base itsSimulation itsBase
)

GENERATE_LIBRARY()
