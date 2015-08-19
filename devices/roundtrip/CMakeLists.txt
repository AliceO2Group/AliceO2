set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/devices/roundtrip
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${ZMQ_INCLUDE_DIR}
  ${AlFa_DIR}/include
)

include_directories(${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

# configure_file(
#   ${CMAKE_SOURCE_DIR}/devices/flp2epn-distributed/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh
# )

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
)

link_directories(${LINK_DIRECTORIES})

set(SRCS
  RoundtripClient.cxx
  RoundtripServer.cxx
)

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
  boost_date_time boost_thread boost_timer boost_system boost_program_options FairMQ
)

set(LIBRARY_NAME RoundtripTest)

GENERATE_LIBRARY()

Set(Exe_Names
  ${Exe_Names}
  rtclient
  rtserver
)

set(Exe_Source
  runRoundtripClient.cxx
  runRoundtripServer.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES RoundtripTest)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
