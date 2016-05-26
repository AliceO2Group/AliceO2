O2_SETUP(NAME common_field)

# TODO get rid of this (true for all modules)
set(DICT_OTHER_MODULES_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/../MathUtils/include
    )

set(SRCS
    src/MagneticWrapperChebyshev.cxx
    src/MagneticField.cxx
    )

set(HEADERS
    include/Field/MagneticWrapperChebyshev.h
    include/Field/MagneticField.h
    )
set(LINKDEF src/FieldLinkDef.h)
set(LIBRARY_NAME Field)
set(BUCKET_NAME common_field_bucket)

O2_GENERATE_LIBRARY()
