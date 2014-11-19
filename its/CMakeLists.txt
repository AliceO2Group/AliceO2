set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/its
${CMAKE_SOURCE_DIR}/header
${BASE_INCLUDE_DIRECTORIES}
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
)

Set(LINKDEF itsLinkDef.h)
Set(LIBRARY_NAME its)
Set(DEPENDENCIES
    Base
)

GENERATE_LIBRARY()
