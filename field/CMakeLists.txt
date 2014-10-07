# Create a library called "libField” which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
${BASE_INCLUDE_DIRECTORIES} 
${ROOT_INCLUDE_DIR}
${CMAKE_SOURCE_DIR}/field 
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
${ROOT_LIBRARY_DIR}
${FAIRROOT_LIBRARY_DIR}

)
 
link_directories( ${LINK_DIRECTORIES})

set(SRCS
AliConstField.cxx   
AliFieldCreator.cxx 
AliFieldPar.cxx
)

set(LINKDEF FieldLinkDef.h)
Set(LIBRARY_NAME Field)
Set(DEPENDENCIES
    Base 
)

GENERATE_LIBRARY()

