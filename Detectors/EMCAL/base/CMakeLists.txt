SET(MODULE_NAME EMCALBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Digit.cxx
  src/EMCGeometry.cxx
  src/Geometry.cxx
  src/Hit.cxx
  src/ShishKebabTrd1Module.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/Digit.h
  include/${MODULE_NAME}/GeometryBase.h
  include/${MODULE_NAME}/Geometry.h
  include/${MODULE_NAME}/EMCGeometry.h
  include/${MODULE_NAME}/Hit.h
  include/${MODULE_NAME}/ShishKebabTrd1Module.h
)

SET(LINKDEF src/EMCALBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_base_bucket)

O2_GENERATE_LIBRARY()
