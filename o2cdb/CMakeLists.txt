configure_file(${CMAKE_SOURCE_DIR}/o2cdb/conditions-server.json ${CMAKE_BINARY_DIR}/bin/config/conditions-server.json)
configure_file(${CMAKE_SOURCE_DIR}/o2cdb/conditions-client.json ${CMAKE_BINARY_DIR}/bin/config/conditions-client.json)
configure_file(${CMAKE_SOURCE_DIR}/o2cdb/fill_local_ocdb.C ${CMAKE_BINARY_DIR}/bin/config/fill_local_ocdb.C)

set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/o2cdb
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${ROOT_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${ZMQ_INCLUDE_DIR}
  ${AlFa_DIR}/include
)

include_directories(${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
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

Set(NO_DICT_SRCS
  ConditionsMQServer.cxx
  ConditionsMQClient.cxx
)

set(DEPENDENCIES
  Base ParBase FairMQ ParMQ ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_LOG_LIBRARY} fairmq_logger pthread Core Tree XMLParser Hist FairTools
)

set(LIBRARY_NAME AliceO2Cdb)
set(LINKDEF O2CdbLinkDef.h)
GENERATE_LIBRARY()

Set(Exe_Names
  conditions-server
  conditions-client
)

Set(Exe_Source
  runConditionsServer.cxx
  runConditionsClient.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  Set(EXE_NAME ${_name})
  Set(SRCS ${_src})
  Set(DEPENDENCIES AliceO2Cdb)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
