set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/simulation/
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
  src/UpgradeGeometryTGeo.cxx
  src/V11Geometry.cxx
  src/UpgradeV1Layer.cxx
  src/Segmentation.cxx
  src/UpgradeSegmentationPixel.cxx
  src/GeometryManager.cxx
  src/Detector.cxx
  src/GeometryHandler.cxx
  src/Point.cxx
  src/Chip.cxx
  src/Digit.cxx
  src/Digitizer.cxx
  src/DigitizerTask.cxx
  src/DigitWriteoutBuffer.cxx
  src/DigitContainer.cxx
  src/DigitLayer.cxx
  src/DigitStave.cxx
)
set(HEADERS
  include/Chip.h
  include/Detector.h
  include/Digit.h
  include/DigitContainer.h
  include/DigitLayer.h
  include/DigitStave.h
  include/DigitWriteoutBuffer.h
  include/Digitizer.h
  include/DigitizerTask.h
  include/GeometryHandler.h
  include/GeometryManager.h
  include/Point.h
  include/Segmentation.h
  include/UpgradeGeometryTGeo.h
  include/UpgradeSegmentationPixel.h
  include/UpgradeV1Layer.h
  include/V11Geometry.h
)

Set(LINKDEF src/itsSimulationLinkDef.h)
Set(LIBRARY_NAME itsSimulation)
Set(DEPENDENCIES
    itsBase
    AliceO2Base
    Base ParBase
)

GENERATE_LIBRARY()
