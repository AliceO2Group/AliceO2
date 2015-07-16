set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/its
${CMAKE_SOURCE_DIR}/header
${BASE_INCLUDE_DIRECTORIES}
${Boost_INCLUDE_DIRS}
${FAIRROOT_INCLUDE_DIR}
${AlFa_DIR}/include
${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})

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
ContainerFactory.cxx
GeometryHandler.cxx
MisalignmentParameter.cxx
Point.cxx
Digit.cxx
Digitizer.cxx
DigitizerTask.cxx
DigitContainer.cxx
DigitLayer.cxx
DigitStave.cxx
)

Set(LINKDEF itsLinkDef.h)
Set(LIBRARY_NAME its)
Set(DEPENDENCIES
    AliceO2Base
)

GENERATE_LIBRARY()
