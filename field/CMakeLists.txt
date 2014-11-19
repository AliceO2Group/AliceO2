# Create a library called "AliceO2Field" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}/field
${CMAKE_SOURCE_DIR}/header
${BASE_INCLUDE_DIRECTORIES}
${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${CMAKE_SOURCE_DIR}/base
${CMAKE_SOURCE_DIR}/field
${FAIRROOT_LIBRARY_DIR}
${ROOT_LIBRARY_DIR}
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
Set(LIBRARY_NAME Field)
Set(DEPENDENCIES Base EG Physics Cint Core)

GENERATE_LIBRARY()

