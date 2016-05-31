set(MODULE_NAME "itsBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/ContainerFactory.cxx
    src/MisalignmentParameter.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/ContainerFactory.h
    include/${MODULE_NAME}/MisalignmentParameter.h
    )


Set(LINKDEF src/itsBaseLinkDef.h)
Set(LIBRARY_NAME itsBase)
set(BUCKET_NAME its_base_bucket)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
