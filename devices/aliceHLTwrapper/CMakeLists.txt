# @author Matthias Richter
# @brief  cmake setup for module devices/aliceHLTwrapper

set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/devices/aliceHLTwrapper
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
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

set(LINK_DIRECTORIES
  ${FAIRROOT_LIBRARY_DIR}
  ${Boost_LIBRARY_DIRS}
)

if(DDS_FOUND)
  set(LINK_DIRECTORIES
   ${LINK_DIRECTORIES}
   ${DDS_LIBRARY_DIR}
  )
endif()

link_directories(${LINK_DIRECTORIES})

set(SRCS
  SystemInterface.cxx
  HOMERFactory.cxx
  WrapperDevice.cxx
  Component.cxx
  MessageFormat.cxx
  EventSampler.cxx
)

if(DDS_FOUND)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    dds-key-value-lib
  )
endif()

set(DEPENDENCIES
    ${DEPENDENCIES}
  )

if(FAIRMQ_DEPENDENCIES)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${FAIRMQ_DEPENDENCIES}
    ${Boost_CHRONO_LIBRARY}
    FairMQ
  )
else(FAIRMQ_DEPENDENCIES)
  set(DEPENDENCIES
    ${DEPENDENCIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_CHRONO_LIBRARY} ${Boost_DATE_TIME_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY} FairMQ
  )
endif(FAIRMQ_DEPENDENCIES)

set(LIBRARY_NAME ALICEHLT)

GENERATE_LIBRARY()

Set(Exe_Names
#  ${Exe_Names}
  aliceHLTWrapper
  aliceHLTEventSampler
  runComponent
)

set(Exe_Source
  aliceHLTWrapper.cxx
  aliceHLTEventSampler.cxx
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
