# Create a library called "AliceO2Field" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_DIR}/field
${CMAKE_SOURCE_DIR}/header
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}
${CMAKE_SOURCE_DIR}/base
${CMAKE_SOURCE_DIR}/field
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
MagneticWrapperChebyshev.cxx
MagneticField.cxx
Chebyshev3D.cxx
Chebyshev3DCalc.cxx
)

Set(HEADERS)
Set(LINKDEF fieldLinkDef.h)
Set(LIBRARY_NAME O2Field)
Set(DEPENDENCIES Base EG Physics Cint Core)

GENERATE_LIBRARY()

