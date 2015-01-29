set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${AlFa_DIR}/include
  ${CMAKE_SOURCE_DIR}/devices/flp2epn-distributed
)

set(DDS_LOCATION $ENV{DDS_LOCATION})

if(DDS_LOCATION)
  message(STATUS "DDS found at ${DDS_LOCATION}")
else()
  message(STATUS "DDS not found")
endif()

if(DDS_LOCATION)
  set(INCLUDE_DIRECTORIES
   ${INCLUDE_DIRECTORIES}
   $ENV{DDS_LOCATION}/include
  )
endif()

include_directories(${INCLUDE_DIRECTORIES})

configure_file(
  ${CMAKE_SOURCE_DIR}/devices/flp2epn-distributed/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh
)

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
)

if(DDS_LOCATION)
  set(LINK_DIRECTORIES
   ${LINK_DIRECTORIES}
   $ENV{DDS_LOCATION}/lib
  )
endif()

link_directories(${LINK_DIRECTORIES})

set(SRCS
  FLPex.cxx
  EPNex.cxx
  FLPexSampler.cxx
)

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
  boost_date_time boost_thread boost_timer boost_system boost_program_options FairMQ
)

if(DDS_LOCATION)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    dds-key-value-lib
  )
endif()

set(LIBRARY_NAME FLP2EPNex_distributed)

GENERATE_LIBRARY()

Set(Exe_Names
  ${Exe_Names}
  testFLP_distributed
  testEPN_distributed
  testFLPSampler
)

if(DDS_LOCATION)
  set(Exe_Names
    ${Exe_Names}
    testFLP_distributed_dds
    testEPN_distributed_dds
    testFLPSampler_dds
  )
endif()

set(Exe_Source
  run/runFLP_distributed.cxx
  run/runEPN_distributed.cxx
  run/runFLPSampler.cxx
)

if(DDS_LOCATION)
  set(Exe_Source
    ${Exe_Source}
    runDDS/runFLP_distributed.cxx
    runDDS/runEPN_distributed.cxx
    runDDS/runFLPSampler.cxx
  )
endif()

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES FLP2EPNex_distributed)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})
