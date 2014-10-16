# Create a library called "libO2its" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}
${BASE_INCLUDE_DIRECTORIES}

${CMAKE_SOURCE_DIR}/AliData
${CMAKE_SOURCE_DIR}/AliBase
${CMAKE_SOURCE_DIR}/its
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

Set(LINKDEF O2itsLinkDef.h)
Set(LIBRARY_NAME its)
Set(DEPENDENCIES
    Base
)

GENERATE_LIBRARY()
