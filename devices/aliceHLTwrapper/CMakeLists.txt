set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/fairmq
  ${CMAKE_SOURCE_DIR}/fairmq/zeromq
  # ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/devices/aliceHLTwrapper
)

include_directories(${INCLUDE_DIRECTORIES})

#configure_file( ${CMAKE_SOURCE_DIR}/example/flp2epn/run/startSomething.sh.in ${CMAKE_BINARY_DIR}/bin/startSomething.sh )

set(LINK_DIRECTORIES
  # ${Boost_LIBRARY_DIRS}
)

link_directories(${LINK_DIRECTORIES})

set(SRCS
  SystemInterface.cxx
  HOMERFactory.cxx
  WrapperDevice.cxx
)

set(DEPENDENCIES
    ${DEPENDENCIES}
    # ${ZMQ_LIBRARY_SHARED}
  )

set(DEPENDENCIES
  ${DEPENDENCIES}
#  ${CMAKE_THREAD_LIBS_INIT}
   FairMQ
)

set(LIBRARY_NAME ALICEHLT)

GENERATE_LIBRARY()

Set(Exe_Names
#  ${Exe_Names}
  aliceHLTWrapper
)

set(Exe_Source 
  aliceHLTWrapper.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES ALICEHLT dl)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
