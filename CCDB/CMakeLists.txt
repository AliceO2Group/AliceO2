configure_file(${CMAKE_SOURCE_DIR}/CCDB/config/conditions-server.json ${CMAKE_BINARY_DIR}/bin/config/conditions-server.json)
configure_file(${CMAKE_SOURCE_DIR}/CCDB/config/conditions-client.json ${CMAKE_BINARY_DIR}/bin/config/conditions-client.json)
configure_file(${CMAKE_SOURCE_DIR}/CCDB/example/fill_local_ocdb.C ${CMAKE_BINARY_DIR}/bin/config/fill_local_ocdb.C)

set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/CCDB/include
  ${CMAKE_SOURCE_DIR}/CCDB

)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${ROOT_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${ZMQ_INCLUDE_DIR}
)

include_directories(${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
)
link_directories(${LINK_DIRECTORIES})

set(SRCS
  src/Manager.cxx
  src/Condition.cxx
  src/GridStorage.cxx
  src/LocalStorage.cxx
  src/FileStorage.cxx
  src/ConditionMetaData.cxx
  src/ConditionId.cxx
  src/IdPath.cxx
  src/IdRunRange.cxx
  src/Storage.cxx
  src/XmlHandler.cxx
)

set(HEADERS
  include/Condition.h
  include/ConditionId.h
  include/ConditionMetaData.h
  include/FileStorage.h
  include/GridStorage.h
  include/IdPath.h
  include/IdRunRange.h
  include/LocalStorage.h
  include/Manager.h
  include/Storage.h
  include/XmlHandler.h
)

Set(NO_DICT_SRCS
  src/ConditionsMQServer.cxx
  src/ConditionsMQClient.cxx
)

set(DEPENDENCIES
  Base ParBase FairMQ ParMQ ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_LOG_LIBRARY} fairmq_logger pthread Core Tree XMLParser Hist
)

set(LIBRARY_NAME CCDB)
set(LINKDEF src/CCDBLinkDef.h)
GENERATE_LIBRARY()

Set(Exe_Names
  conditions-server
  conditions-client
)

Set(Exe_Source
  src/runConditionsServer.cxx
  src/runConditionsClient.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  Set(EXE_NAME ${_name})
  Set(SRCS ${_src})
  Set(DEPENDENCIES CCDB)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
