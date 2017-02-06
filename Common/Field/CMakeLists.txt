set(MODULE_NAME "Field")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/MagneticWrapperChebyshev.cxx
    src/MagFieldParam.cxx
    src/MagneticField.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/MagneticWrapperChebyshev.h
    include/${MODULE_NAME}/MagneticField.h
    include/${MODULE_NAME}/MagFieldParam.h
    )
set(LINKDEF src/FieldLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME common_field_bucket)

O2_GENERATE_LIBRARY()
