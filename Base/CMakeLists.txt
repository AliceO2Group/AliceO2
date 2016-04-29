set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Base
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
  ${CMAKE_SOURCE_DIR}/Base
  ${FAIRROOT_LIBRARY_DIR}
  ${ROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
  src/Detector.cxx
  src/TrackReference.cxx
)

Set(HEADERS
  include/TrackReference.h
  include/Detector.h
)
Set(LINKDEF src/BaseLinkDef.h)
Set(LIBRARY_NAME AliceO2Base)
Set(DEPENDENCIES Base parbase EG Physics Core )

GENERATE_LIBRARY()
