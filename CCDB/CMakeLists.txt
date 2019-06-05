set(MODULE_NAME "CCDB")

O2_SETUP(NAME ${MODULE_NAME})

# When Protobuf is built with CMake, its PROTOBUF_GENERATE_CPP always puts the output files
# in ${CMAKE_CURRENT_BINARY_DIR} but sets the path strings to ${CMAKE_CURRENT_BINARY_DIR}/src
# which is not what we want.
# The following function allows a destination path as argument, which we'll set to empty.
function(PROTOBUF_GENERATE_CPP_DEST PATH SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(DIR ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.h")

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${PATH})

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}/${PATH} ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

PROTOBUF_GENERATE_CPP_DEST("" PROTO_SRCS PROTO_HDRS src/request.proto)

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
  src/CcdbApi.cxx
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
  o2-ccdb-conditions-server
  o2-ccdb-conditions-client
  o2-ccdb-standalone-client
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

if (NOT APPLE)
  set(TEST_SRCS
     test/testWriteReadAny.cxx
     test/testCcdbApi.cxx
  )

  O2_GENERATE_TESTS(
    BUCKET_NAME ${BUCKET_NAME}
    MODULE_LIBRARY_NAME ${MODULE_NAME}
    TEST_SRCS ${TEST_SRCS}
  )
endif()
