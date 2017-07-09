set(MODULE_NAME "Field")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/MagneticWrapperChebyshev.cxx
    src/MagneticField.cxx
    src/MagFieldParam.cxx
    src/MagFieldContFact.cxx
    src/MagFieldFast.cxx
    src/MagFieldFact.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/MagneticWrapperChebyshev.h
    include/${MODULE_NAME}/MagneticField.h
    include/${MODULE_NAME}/MagFieldParam.h
    include/${MODULE_NAME}/MagFieldContFact.h
    include/${MODULE_NAME}/MagFieldFast.h
    include/${MODULE_NAME}/MagFieldFact.h
    )
set(LINKDEF src/FieldLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME common_field_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testMagneticField.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
