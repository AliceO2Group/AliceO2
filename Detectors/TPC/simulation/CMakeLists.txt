
set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/include

)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
)
include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Detector.cxx
   src/Point.cxx
   src/DigitizerTask.cxx
   src/Digitizer.cxx
   src/DigitContainer.cxx
   src/DigitCRU.cxx
   src/DigitRow.cxx
   src/DigitPad.cxx
   src/DigitTime.cxx
   src/DigitADC.cxx
   src/Digit.cxx
   src/Cluster.cxx
   src/BoxCluster.cxx
   src/ClusterContainer.cxx
   src/BoxClusterer.cxx
   src/ClustererTask.cxx
)

set(HEADERS
   include/Detector.h
   include/Point.h
   include/DigitizerTask.h
   include/Digitizer.h
   include/DigitContainer.h
   include/DigitCRU.h
   include/DigitRow.h
   include/DigitPad.h
   include/DigitTime.h
   include/DigitADC.h
   include/Digit.h
   include/Cluster.h
   include/BoxCluster.h
   include/ClusterContainer.h
   include/BoxClusterer.h
   include/ClustererTask.h
)
Set(LINKDEF src/tpcSimulationLinkDef.h)
Set(LIBRARY_NAME tpcSimulation)
Set(DEPENDENCIES
    tpcBase
)
GENERATE_LIBRARY()
