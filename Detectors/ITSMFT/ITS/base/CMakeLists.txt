set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
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
  src/ContainerFactory.cxx
  src/MisalignmentParameter.cxx
)

set(HEADERS
  include/ContainerFactory.h
  include/MisalignmentParameter.h
)


Set(LINKDEF src/itsBaseLinkDef.h)
Set(LIBRARY_NAME itsBase)
Set(DEPENDENCIES
    AliceO2Base ParBase 
)

GENERATE_LIBRARY()
