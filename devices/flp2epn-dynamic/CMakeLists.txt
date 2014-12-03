set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES} 
  ${Boost_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${AlFa_DIR}/include
  ${CMAKE_SOURCE_DIR}/devices/flp2epn-dynamic
)

include_directories(${INCLUDE_DIRECTORIES})

configure_file( ${CMAKE_SOURCE_DIR}/devices/flp2epn-dynamic/run/startFLP2EPN-dynamic.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-dynamic.sh )

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
)

link_directories(${LINK_DIRECTORIES})

set(SRCS
  O2FLPex.cxx
  O2EPNex.cxx
)

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
  boost_date_time boost_thread boost_timer boost_system boost_program_options FairMQ
)

set(LIBRARY_NAME FLP2EPNex_dynamic)

GENERATE_LIBRARY()

Set(Exe_Names
  ${Exe_Names}
  testFLP_dynamic
  testEPN_dynamic
)

set(Exe_Source 
  run/runFLP_dynamic.cxx
  run/runEPN_dynamic.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES FLP2EPNex_dynamic)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
