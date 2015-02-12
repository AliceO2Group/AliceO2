set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${ZMQ_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/devices/aliceHLTwrapper
  ${FAIRROOT_INCLUDE_DIR}
  ${AlFa_DIR}/include
)

include_directories(${INCLUDE_DIRECTORIES})
# set(CXXOPT "-O0 -g -ggdb")
# set(CXXFLAGS "-O0 -g -ggdb")

#configure_file( ${CMAKE_SOURCE_DIR}/example/flp2epn/run/startSomething.sh.in ${CMAKE_BINARY_DIR}/bin/startSomething.sh )

set(LINK_DIRECTORIES
  ${FAIRROOT_LIBRARY_DIR}
  ${Boost_LIBRARY_DIRS}
  ${AlFa_DIR}/lib
)

link_directories(${LINK_DIRECTORIES})

set(SRCS
  SystemInterface.cxx
  HOMERFactory.cxx
  WrapperDevice.cxx
  Component.cxx
  MessageFormat.cxx
)

set(DEPENDENCIES
    ${DEPENDENCIES}
    # ${ZMQ_LIBRARY_SHARED}
  )

set(DEPENDENCIES
  ${DEPENDENCIES}
#  ${CMAKE_THREAD_LIBS_INIT}
   boost_thread boost_system FairMQ
)

set(LIBRARY_NAME ALICEHLT)

GENERATE_LIBRARY()

Set(Exe_Names
#  ${Exe_Names}
  aliceHLTWrapper
  runComponent
)

set(Exe_Source
  aliceHLTWrapper.cxx
  runComponent.cxx
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

