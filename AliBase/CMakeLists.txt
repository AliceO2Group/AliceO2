# Create a library called "O2MCStack" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR} 
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_DIR}/AliBase
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}
${CMAKE_SOURCE_DIR}/AliBase

)
 
link_directories( ${LINK_DIRECTORIES})

set(SRCS
O2Module.cxx
O2Detector.cxx
O2TrackReference.cxx
)

Set(HEADERS)
Set(LINKDEF O2BaseLinkDef.h)
Set(LIBRARY_NAME O2Base)
Set(DEPENDENCIES Base EG Physics Cint Core)

GENERATE_LIBRARY()

