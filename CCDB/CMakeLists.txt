O2_SETUP(NAME "CCDB")

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
  include/CCDB/Condition.h
  include/CCDB/ConditionId.h
  include/CCDB/ConditionMetaData.h
  include/CCDB/FileStorage.h
  include/CCDB/GridStorage.h
  include/CCDB/IdPath.h
  include/CCDB/IdRunRange.h
  include/CCDB/LocalStorage.h
  include/CCDB/Manager.h
  include/CCDB/Storage.h
  include/CCDB/XmlHandler.h
)

Set(NO_DICT_SRCS
  src/ConditionsMQServer.cxx
  src/ConditionsMQClient.cxx
)

set(LIBRARY_NAME CCDB)
set(LINKDEF src/CCDBLinkDef.h)
set(BUCKET_NAME CCDB_Bucket)
O2_GENERATE_LIBRARY()

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

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(LIBRARY_NAME CCDB)
set(BUCKET_NAME CCDB_Bucket)

foreach (_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  Set(EXE_NAME ${_name})
  Set(SRCS ${_src})
  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

install(
    FILES config/conditions-server.json
    config/conditions-client.json
    example/fill_local_ocdb.C
    DESTINATION bin/config
)