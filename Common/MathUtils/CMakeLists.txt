set(MODULE_NAME "MathUtils")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Chebyshev3D.cxx
  src/Chebyshev3DCalc.cxx
  src/MathBase.cxx
  src/Cartesian2D.cxx
  src/Cartesian3D.cxx
  src/CachingTF1.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Chebyshev3D.h
  include/${MODULE_NAME}/Chebyshev3DCalc.h
  include/${MODULE_NAME}/MathBase.h
  include/${MODULE_NAME}/Cartesian2D.h
  include/${MODULE_NAME}/Cartesian3D.h
  include/${MODULE_NAME}/CachingTF1.h
)

set(LINKDEF src/MathUtilsLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME common_math_bucket)

O2_GENERATE_LIBRARY()


set(TEST_SRCS
  test/testCartesian3D.cxx
  test/testCachingTF1.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
