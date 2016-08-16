add_subdirectory(QCCommon)

# We add it explicitely because it is a header only module.
include_directories(QCCommon/include)

add_subdirectory(QCProducer)
add_subdirectory(QCMerger)
add_subdirectory(QCViewer)
#Set(INCLUDE_DIRECTORIES
#  ${ROOT_INCLUDE_DIR}
#  ${BASE_INCLUDE_DIRECTORIES}
#  ${FAIRROOT_INCLUDE_DIR}
#  ${CMAKE_SOURCE_DIR}/Utilities/QC
#  ${CMAKE_SOURCE_DIR}/Utilities/QC/Merger
#  ${CMAKE_SOURCE_DIR}/Utilities/QC/Producer
#  ${CMAKE_SOURCE_DIR}/Utilities/QC/Viewer
#)
#
#Set(SYSTEM_INCLUDE_DIRECTORIES
#  ${Boost_INCLUDE_DIR}
#  ${ZMQ_INCLUDE_DIR}
#)
#
#Include_Directories(${INCLUDE_DIRECTORIES})
#Include_Directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})
#
#set(LINK_DIRECTORIES
#  ${Boost_LIBRARY_DIRS}
#  ${FAIRROOT_LIBRARY_DIR}
#  ${ROOT_LIBRARY_DIR}
#)
#
#Link_Directories(${LINK_DIRECTORIES})
#
#set(SRCS
#    Producer/src/ProducerDevice.cxx
#    Producer/src/HistogramProducer.cxx
#    Producer/src/TreeProducer.cxx
#    Merger/src/MergerDevice.cxx
#    Merger/src/Merger.cxx
#    Viewer/src/ViewerDevice.cxx
#)
#
#set(DEPENDENCIES
#  ${CMAKE_THREAD_LIBS_INIT}
#  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
#  RIO
#  Core
#  MathMore
#  Net
#  Hist
#  Tree
#  Gpad
#)
#
#set(LIBRARY_NAME o2qaLibrary)
#GENERATE_LIBRARY()
#
#Set(Exe_Names
#  runMergerDevice
#  runProducer
#  runViewerDevice
#)
#
#Set(Exe_Source
#    Merger/src/runMerger.cxx
#    Producer/src/runProducer.cxx
#    Viewer/src/runViewerDevice.cxx
#)
#
#list(LENGTH Exe_Names _length)
#math(EXPR _length ${_length}-1)
#
#ForEach(_file RANGE 0 ${_length})
#  list(GET Exe_Names ${_file} _name)
#  list(GET Exe_Source ${_file} _src)
#  set(EXE_NAME ${_name})
#  set(SRCS ${_src})
#  set(DEPENDENCIES dl Core Base Hist o2qaLibrary FairMQ pthread fairmq_logger ${Boost_SYSTEM_LIBRARY} ${Boost_LOG_LIBRARY} )
#  GENERATE_EXECUTABLE()
#EndForEach(_file RANGE 0 ${_length})
#
#ENABLE_TESTING()
#ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
#SET(EXECUTABLE_PATH ${CMAKE_BINARY_DIR}/bin)
#
#add_executable(testQaProducer Producer/test/ProducerTestSuite.cxx)
#add_executable(testQaMerger Merger/test/MergerTestSuite.cxx)
#add_executable(testQaViewer Viewer/test/ViewerTestSuite.cxx)
#
#target_link_libraries(testQaProducer dl Core Base Hist o2qaLibrary FairMQ  ${Boost_SYSTEM_LIBRARY} )
#target_link_libraries(testQaMerger dl Core Base Hist o2qaLibrary FairMQ  ${Boost_SYSTEM_LIBRARY} )
#target_link_libraries(testQaViewer dl Core Base Hist o2qaLibrary FairMQ ${Boost_SYSTEM_LIBRARY} )
#
#add_test(ProducerTest ${EXECUTABLE_PATH}/testQaProducer)
#add_test(MergerTest ${EXECUTABLE_PATH}/testQaMerger)
#add_test(ViewerTest ${EXECUTABLE_PATH}/testQaViewer)
