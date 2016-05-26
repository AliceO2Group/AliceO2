configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run/startFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN-distributed.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}//test/testFLP2EPN-distributed.sh.in ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)

#set(INCLUDE_DIRECTORIES
#  ${CMAKE_SOURCE_DIR}/Examples/flp2epn-distributed
#)
#
#set(SYSTEM_INCLUDE_DIRECTORIES
#  ${BASE_INCLUDE_DIRECTORIES}
#  ${Boost_INCLUDE_DIR}
#  ${ZMQ_INCLUDE_DIR}
#  ${FAIRROOT_INCLUDE_DIR}
#  ${AlFa_DIR}/include
#)

O2_SETUP(NAME "flp2epn-distributed")

#if(DDS_FOUND)
#  add_definitions(-DENABLE_DDS)
#
##  set(SYSTEM_INCLUDE_DIRECTORIES
##   ${SYSTEM_INCLUDE_DIRECTORIES}
##   ${DDS_INCLUDE_DIR}
##  )
#endif()

#include_directories(${INCLUDE_DIRECTORIES})
#include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

#set(LINK_DIRECTORIES
#  ${Boost_LIBRARY_DIRS}
#  ${FAIRROOT_LIBRARY_DIR}
#  ${AlFa_DIR}/lib
#)

#if(DDS_FOUND)
#  set(LINK_DIRECTORIES
#   ${LINK_DIRECTORIES}
#   ${DDS_LIBRARY_DIR}
#  )
#endif()

#link_directories(${LINK_DIRECTORIES})

set(SRCS
    src/FLPSyncSampler.cxx
    src/FLPSender.cxx
    src/EPNReceiver.cxx
    )

set(HEADERS
    include/FLPSyncSampler.h
    include/FLPSender.h
    include/EPNReceiver.h)

#if(FAIRMQ_DEPENDENCIES)
#  set(DEPENDENCIES
#    ${DEPENDENCIES}
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${FAIRMQ_DEPENDENCIES}
#    FairMQ
#  )
#else(FAIRMQ_DEPENDENCIES)
#  set(DEPENDENCIES
#    ${DEPENDENCIES}
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${Boost_DATE_TIME_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_CHRONO_LIBRARY} FairMQ
#  )
#endif(FAIRMQ_DEPENDENCIES)

#if(DDS_FOUND)
#  set(DEPENDENCIES
#    ${DEPENDENCIES}
#    dds-key-value-lib
#  )
#endif()

if (DDS_FOUND)
  set(BUCKET_NAME flp2epndistrib_Bucket)
else ()
  set(BUCKET_NAME flp2epn_Bucket)
endif ()
set(LIBRARY_NAME FLP2EPNex_distributed)

O2_GENERATE_LIBRARY()

Set(Exe_Names
    ${Exe_Names}
    flpSyncSampler
    flpSender
    epnReceiver
    )

if (DDS_FOUND)
  set(Exe_Names
      ${Exe_Names}
      flpSyncSampler_dds
      flpSender_dds
      epnReceiver_dds
      )
endif ()

set(Exe_Source
    run/runFLPSyncSampler.cxx
    run/runFLPSender.cxx
    run/runEPNReceiver.cxx
    )

if (DDS_FOUND)
  set(Exe_Source
      ${Exe_Source}
      runO2Prototype/runFLPSyncSampler.cxx
      runO2Prototype/runFLPSender.cxx
      runO2Prototype/runEPNReceiver.cxx
      )
endif ()

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
if (DDS_FOUND)
  set(BUCKET_NAME flp2epndistrib_Bucket)
else ()
  set(BUCKET_NAME flp2epn_Bucket)
endif ()
set(LIBRARY_NAME FLP2EPNex_distributed)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach (_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
EndForEach (_file RANGE 0 ${_length})

add_test(NAME run_flp2epn_distributed COMMAND ${CMAKE_BINARY_DIR}/Examples/flp2epn-distributed/test/testFLP2EPN-distributed.sh)
set_tests_properties(run_flp2epn_distributed PROPERTIES TIMEOUT "30")
set_tests_properties(run_flp2epn_distributed PROPERTIES PASS_REGULAR_EXPRESSION "acknowledged after")
