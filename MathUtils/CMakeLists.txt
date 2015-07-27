# Create a library called "MathUtils" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}/MathUtils
)

set(SYSTEM_INCLUDE_DIRECTORIES
${CMAKE_SOURCE_DIR}/MathUtils
${BASE_INCLUDE_DIRECTORIES}
${FAIRROOT_INCLUDE_DIR}
${AlFa_DIR}/include
${ROOT_INCLUDE_DIR}
)

include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${CMAKE_SOURCE_DIR}/MathUtils
${FAIRROOT_LIBRARY_DIR}
${ROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
Chebyshev3D.cxx
Chebyshev3DCalc.cxx
)

Set(HEADERS)
Set(LINKDEF MathUtilsLinkDef.h)
Set(LIBRARY_NAME MathUtils)
Set(DEPENDENCIES Core)

GENERATE_LIBRARY()

