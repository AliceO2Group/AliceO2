SET(MODULE_NAME EMCALBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Digit.cxx
  src/Hit.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Digit.h
  include/${MODULE_NAME}/Hit.h
)

SET(LINKDEF src/EMCALBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_base_bucket)

O2_GENERATE_LIBRARY()
