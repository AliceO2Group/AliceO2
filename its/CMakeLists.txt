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
AliITSUGeomTGeo.cxx
AliITSv11Geometry.cxx
AliITSUv1Layer.cxx
AliITSsegmentation.cxx
AliITSUSegmentationPix.cxx
AliGeomManager.cxx
O2its.cxx
O2itsContFact.cxx
O2itsGeoHandler.cxx
O2itsMisalignPar.cxx
O2itsPoint.cxx
O2itsDigi.cxx
O2itsDigiTask.cxx
)

Set(LINKDEF O2itsLinkDef.h)
Set(LIBRARY_NAME its)
Set(DEPENDENCIES
    Base
)

GENERATE_LIBRARY()
