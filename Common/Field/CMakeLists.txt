set(MODULE_NAME "Field")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/MagneticWrapperChebyshev.cxx
    src/MagneticField.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/MagneticWrapperChebyshev.h
    include/${MODULE_NAME}/MagneticField.h
    )
set(LINKDEF src/FieldLinkDef.h)
set(LIBRARY_NAME Field)
set(BUCKET_NAME common_field_bucket)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
