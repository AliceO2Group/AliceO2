set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES} 
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/o2cdb
  ${ROOT_INCLUDE_DIR}
)
include_directories(${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
)
link_directories(${LINK_DIRECTORIES})

set(SRCS
  Manager.cxx
  Condition.cxx
  GridStorage.cxx
  LocalStorage.cxx
  FileStorage.cxx
  ConditionMetaData.cxx
  ConditionId.cxx
  IdPath.cxx
  IdRunRange.cxx
  Storage.cxx
  XmlHandler.cxx
)

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
  boost_thread boost_system FairMQ
)

set(LIBRARY_NAME AliceO2Cdb)
Set(LINKDEF O2CdbLinkDef.h)
GENERATE_LIBRARY()
