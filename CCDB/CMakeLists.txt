set(MODULE_NAME "CCDB")

O2_SETUP(NAME ${MODULE_NAME})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS src/request.proto)
include_directories(${CMAKE_BINARY_DIR}/CCDB)

set(SRCS
  src/Backend.cxx
  src/BackendOCDB.cxx
  src/BackendRiak.cxx
  src/Condition.cxx
  src/ConditionId.cxx
  src/ConditionMetaData.cxx
  src/FileStorage.cxx
  src/GridStorage.cxx
  src/IdPath.cxx
  src/IdRunRange.cxx
  src/LocalStorage.cxx
  src/Manager.cxx
  src/ObjectHandler.cxx
  src/Storage.cxx
  src/XmlHandler.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Backend.h
  include/${MODULE_NAME}/BackendOCDB.h
  include/${MODULE_NAME}/BackendRiak.h
  include/${MODULE_NAME}/Condition.h
  include/${MODULE_NAME}/ConditionId.h
  include/${MODULE_NAME}/ConditionMetaData.h
  include/${MODULE_NAME}/FileStorage.h
  include/${MODULE_NAME}/GridStorage.h
  include/${MODULE_NAME}/IdPath.h
  include/${MODULE_NAME}/IdRunRange.h
  include/${MODULE_NAME}/LocalStorage.h
  include/${MODULE_NAME}/Manager.h
  include/${MODULE_NAME}/ObjectHandler.h
  include/${MODULE_NAME}/Storage.h
  include/${MODULE_NAME}/XmlHandler.h
  test/TestClass.h
)

Set(NO_DICT_SRCS
  src/ConditionsMQServer.cxx
  src/ConditionsMQClient.cxx
  ${PROTO_SRCS}
)

set(LIBRARY_NAME ${MODULE_NAME})

set(LINKDEF src/CCDBLinkDef.h)
set(BUCKET_NAME CCDB_bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
  conditions-server
  conditions-client
  standalone-client
)

Set(Exe_Source
  src/runConditionsServer.cxx
  src/runConditionsClient.cxx
  test/testQueryServerStandalone.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
#  Set(DEPENDENCIES CCDB)
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

set(TEST_SRCS
   test/testWriteReadAny.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)


