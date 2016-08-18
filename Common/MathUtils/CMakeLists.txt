set(MODULE_NAME "MathUtils")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Chebyshev3D.cxx
  src/Chebyshev3DCalc.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Chebyshev3D.h
  include/${MODULE_NAME}/Chebyshev3DCalc.h
)
set(LINKDEF src/MathUtilsLinkDef.h)
set(LIBRARY_NAME MathUtils)
set(BUCKET_NAME common_math_bucket)

O2_GENERATE_LIBRARY()
