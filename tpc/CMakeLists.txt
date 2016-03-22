# Create a library called "libO2tpc" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}/
${CMAKE_SOURCE_DIR}/tpc
${CMAKE_SOURCE_DIR}/tpc/dirty
${CMAKE_SOURCE_DIR}/o2cdb
)

set(SYSTEM_INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}
${BASE_INCLUDE_DIRECTORIES}
${Boost_INCLUDE_DIRS}
${FAIRROOT_INCLUDE_DIR}
${AlFa_DIR}/include
)

include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${Boost_LIBRARY_DIRS}
${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
ContainerFactory.cxx
Point.cxx
Detector.cxx
dirty/AliDetectorParam.cxx
dirty/AliH2F.cxx
dirty/AliMathBase.cxx
dirty/AliTPCParam.cxx
dirty/AliTPCParamSR.cxx
dirty/AliTPCPRF2D.cxx
dirty/AliTPCRF1D.cxx
dirty/AliTPCROC.cxx
dirty/AliLog.cxx
)

Set(LINKDEF TpcLinkDef.h)
Set(LIBRARY_NAME tpc)
Set(DEPENDENCIES
    Base
    Minuit boost_log
    AliceO2Base
    AliceO2Cdb
)
GENERATE_LIBRARY()
