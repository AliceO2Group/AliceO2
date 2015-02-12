# Create a library called "libO2tpc" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}
${BASE_INCLUDE_DIRECTORIES}
${Boost_INCLUDE_DIRS}
${FAIRROOT_INCLUDE_DIR}
${AlFa_DIR}/include
#put here all directories where header files are located
${CMAKE_SOURCE_DIR}/
${CMAKE_SOURCE_DIR}/tpc
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
ContainerFactory.cxx
Point.cxx
Detector.cxx
)

Set(LINKDEF TpcLinkDef.h)
Set(LIBRARY_NAME tpc)
Set(DEPENDENCIES
    Base
)
GENERATE_LIBRARY()
