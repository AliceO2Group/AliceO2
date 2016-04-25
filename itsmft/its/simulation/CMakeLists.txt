set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/itsmft/its/simulation/include
  ${CMAKE_SOURCE_DIR}/itsmft/its/simulation/src
  ${CMAKE_SOURCE_DIR}/itsmft/its/base/include
  ${CMAKE_SOURCE_DIR}/itsmft/its/base/src
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
  ${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
  UpgradeGeometryTGeo.cxx
  V11Geometry.cxx
  UpgradeV1Layer.cxx
  Segmentation.cxx
  UpgradeSegmentationPixel.cxx
  GeometryManager.cxx
  Detector.cxx
  GeometryHandler.cxx
  Point.cxx
  Chip.cxx
  Digit.cxx
  Digitizer.cxx
  DigitizerTask.cxx
  DigitWriteoutBuffer.cxx
  DigitContainer.cxx
  DigitLayer.cxx
  DigitStave.cxx
)
set(HEADERS
  UpgradeGeometryTGeo.h
  V11Geometry.h
  UpgradeV1Layer.h
  Segmentation.h
  UpgradeSegmentationPixel.h
  GeometryManager.h
  Detector.h
  GeometryHandler.h
  Point.h
  Chip.h
  Digit.h
  Digitizer.h
  DigitizerTask.h
  DigitWriteoutBuffer.h
  DigitContainer.h
  DigitLayer.h
  DigitStave.h
)

Set(LINKDEF src/itsSimualationLinkDef.h)
Set(LIBRARY_NAME itsSimulation)
Set(DEPENDENCIES
    itsBase
    AliceO2Base
)

GENERATE_LIBRARY()
