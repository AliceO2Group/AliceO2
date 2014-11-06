# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR}
${PYTHIA8_INCLUDE_DIR}
${SIMPATH}/include
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_DIR}/Generators
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}
${SIMPATH}/lib
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
Pythia6Generator.cxx
Pythia8Generator.cxx
)

set(LINKDEF GeneratorsLinkDef.h)
set(LIBRARY_NAME AliGen)

set(DEPENDENCIES Base AliceO2Data pythia8 Pythia6)

GENERATE_LIBRARY()
