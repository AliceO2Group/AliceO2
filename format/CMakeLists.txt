# @author Matthias Richter
# @brief  cmake setup for the test of format implementation

set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/format
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIR}
  ${FAIRROOT_INCLUDE_DIR}
)

include_directories(${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${FAIRROOT_LIBRARY_DIR}
  ${Boost_LIBRARY_DIRS}
)

link_directories(${LINK_DIRECTORIES})

#library source
set(SRCS
)

set(DEPENDENCIES
    ${DEPENDENCIES}
  )

set(DEPENDENCIES
  ${DEPENDENCIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

set(SRCS
  message_list.cxx
)

set(LIBRARY_NAME aliceo2format)

GENERATE_LIBRARY()

Set(Exe_Names
  testFormat
)

set(Exe_Source
  testFormat.cxx
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
