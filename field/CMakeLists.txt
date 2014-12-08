# Create a library called "AliceO2Field" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/field
${CMAKE_SOURCE_DIR}/header
${BASE_INCLUDE_DIRECTORIES}
${FAIRROOT_INCLUDE_DIR}
${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${FAIRROOT_LIBRARY_DIR}
${ROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
MagneticWrapperChebyshev.cxx
MagneticField.cxx
)

Set(HEADERS)
Set(LINKDEF fieldLinkDef.h)
Set(LIBRARY_NAME Field)
Set(DEPENDENCIES MathUtils Cint Core)

GENERATE_LIBRARY()

