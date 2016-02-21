Set(INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}  
  ${BASE_INCLUDE_DIRECTORIES} 
  ${FAIRROOT_INCLUDE_DIR}
  ${AlFa_DIR}/include
  ${CMAKE_SOURCE_DIR}/o2qa
  ${CMAKE_SOURCE_DIR}/o2qa/Merger
  ${CMAKE_SOURCE_DIR}/o2qa/Producer
  ${CMAKE_SOURCE_DIR}/o2qa/Viewer
#  ${CMAKE_SOURCE_DIR}/o2qa/SystemController
)

Set(SYSTEM_INCLUDE_DIRECTORIES
  ${Boost_INCLUDE_DIR}
)

Include_Directories(${INCLUDE_DIRECTORIES})
Include_Directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${Boost_LIBRARY_DIRS}
  ${FAIRROOT_LIBRARY_DIR}
  ${AlFa_DIR}/lib
  ${SIMPATH}/lib/root
)

Link_Directories(${LINK_DIRECTORIES})

set(SRCS
  Producer/ProducerDevice.cxx
  Producer/HistogramProducer.cxx
  Producer/TreeProducer.cxx
  Merger/MergerDevice.cxx
  Merger/Merger.cxx
  Viewer/HistogramViewer.cxx
#  SystemController/SystemController.cxx
)

set(DEPENDENCIES
  ${CMAKE_THREAD_LIBS_INIT}
  boost_unit_test_framework
)

set(LIBRARY_NAME o2qaLibrary)
GENERATE_LIBRARY()

Set(Exe_Names
  runMergerDevice
  runProducer
  runHistogramViewer
#  runSystemController
)

Set(Exe_Source
  runMerger.cxx
  runProducer.cxx
  runHistogramViewer.cxx
#  runSystemController.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  set(EXE_NAME ${_name})
  set(SRCS ${_src})
  set(DEPENDENCIES dl Core Base Hist o2qaLibrary FairMQ)
  GENERATE_EXECUTABLE()
EndForEach(_file RANGE 0 ${_length})

ENABLE_TESTING()
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
SET(EXECUTABLE_PATH ${CMAKE_SOURCE_DIR}/build_o2/bin)

add_executable(testQaProducer Producer/Tests/ProducerTestSuite.cxx)
add_executable(testQaMerger Merger/Tests/MergerTestSuite.cxx)

target_link_libraries(testQaProducer dl Core Base Hist o2qaLibrary FairMQ)
target_link_libraries(testQaMerger dl Core Base Hist o2qaLibrary FairMQ)

add_test(ProducerTest ${EXECUTABLE_PATH}/testQaProducer)
add_test(MergerTest ${EXECUTABLE_PATH}/testQaMerger)
