 # Create a library called "libAliData" 

set(INCLUDE_DIRECTORIES
${ROOT_INCLUDE_DIR} 
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_DIR}/AliData
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}
${CMAKE_SOURCE_DIR}/AliData

)
 
link_directories( ${LINK_DIRECTORIES})

set(SRCS
AliStack.cxx 
AliMCTrack.cxx
)

Set(HEADERS )
Set(LINKDEF MCStackLinkDef.h)
Set(LIBRARY_NAME AliData)
Set(DEPENDENCIES Base EG Physics Cint Core)

GENERATE_LIBRARY()

