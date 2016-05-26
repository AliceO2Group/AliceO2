O2_SETUP(NAME MathUtils)

set(SRCS
  src/Chebyshev3D.cxx
  src/Chebyshev3DCalc.cxx
)

set(HEADERS
  include/MathUtils/Chebyshev3D.h
  include/MathUtils/Chebyshev3DCalc.h
)
set(LINKDEF src/MathUtilsLinkDef.h)
set(LIBRARY_NAME MathUtils)
set(BUCKET_NAME common_math_bucket)

O2_GENERATE_LIBRARY()
