set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/devices/flp2epn-distributed
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${ZMQ_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
  ${AlFa_DIR}/include
)

if(DDS_FOUND)
  add_definitions(-DENABLE_DDS)

  set(SYSTEM_INCLUDE_DIRECTORIES
   ${SYSTEM_INCLUDE_DIRECTORIES}
   ${DDS_INCLUDE_DIR}
  )
endif()

include_directories(${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

configure_file(
  ${CMAKE_SOURCE_DIR}/devices/flp2epn-distributed/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh
)

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
)

if(DDS_FOUND)
  set(LINK_DIRECTORIES
   ${LINK_DIRECTORIES}
   ${DDS_LIBRARY_DIR}
  )
endif()

link_directories(${LINK_DIRECTORIES})

set(SRCS
  FLPSyncSampler.cxx
  FLPSender.cxx
  EPNReceiver.cxx
)

if(FAIRMQ_DEPENDENCIES)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${FAIRMQ_DEPENDENCIES}
    FairMQ
  )
else(FAIRMQ_DEPENDENCIES)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    boost_date_time boost_thread boost_timer boost_system boost_program_options boost_chrono FairMQ
  )
endif(FAIRMQ_DEPENDENCIES)

if(DDS_FOUND)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    dds-key-value-lib
  )
endif()

set(LIBRARY_NAME FLP2EPNex_distributed)

GENERATE_LIBRARY()

Set(Exe_Names
  ${Exe_Names}
  flpSyncSampler
  flpSender
  epnReceiver
)

if(DDS_FOUND)
  set(Exe_Names
    ${Exe_Names}
    flpSyncSampler_dds
    flpSender_dds
    epnReceiver_dds
  )
endif()

set(Exe_Source
  run/runFLPSyncSampler.cxx
  run/runFLPSender.cxx
  run/runEPNReceiver.cxx
)

if(DDS_FOUND)
  set(Exe_Source
    ${Exe_Source}
    runO2Prototype/runFLPSyncSampler.cxx
    runO2Prototype/runFLPSender.cxx
    runO2Prototype/runEPNReceiver.cxx
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
