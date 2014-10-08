set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES} 
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/devices/flp2epn
)

include_directories(${INCLUDE_DIRECTORIES})

configure_file( ${CMAKE_SOURCE_DIR}/devices/flp2epn/run/startFLP2EPN.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN.sh )
configure_file( ${CMAKE_SOURCE_DIR}/devices/flp2epn/run/startMerger.sh.in ${CMAKE_BINARY_DIR}/bin/startMerger.sh )

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
)

link_directories(${LINK_DIRECTORIES})

set(SRCS
  O2FLPex.cxx
  O2EPNex.cxx
  O2Proxy.cxx
  O2Merger.cxx
  O2EpnMerger.cxx
)

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
  boost_thread boost_timer boost_system FairMQ
)

set(LIBRARY_NAME FLP2EPNex)

GENERATE_LIBRARY()

Set(Exe_Names
  ${Exe_Names}
  testFLP
  testEPN
  testEPN_M
  testProxy
  testMerger
)

set(Exe_Source 
  run/runFLP.cxx
  run/runEPN.cxx
  run/runEPN_M.cxx
  run/runProxy.cxx
  run/runMerger.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES FLP2EPNex)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
