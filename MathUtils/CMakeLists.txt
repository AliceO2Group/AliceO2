# Create a library called "MathUtils" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/MathUtils/include
)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/MathUtils
  ${BASE_INCLUDE_DIRECTORIES}
  ${FAIRROOT_INCLUDE_DIR}
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
  src/Chebyshev3D.cxx
  src/Chebyshev3DCalc.cxx
)

Set(HEADERS
  include/Chebyshev3D.h
  include/Chebyshev3DCalc.h
)
Set(LINKDEF src/MathUtilsLinkDef.h)
Set(LIBRARY_NAME MathUtils)
Set(DEPENDENCIES Core)

GENERATE_LIBRARY()
