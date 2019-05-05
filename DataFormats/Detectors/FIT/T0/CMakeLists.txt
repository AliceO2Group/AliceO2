set(MODULE_NAME "DataFormatsFITT0")
set(MODULE_BUCKET_NAME data_format_fit_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Digit.cxx
  src/RecPoints.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Digit.h
    include/${MODULE_NAME}/RecPoints.h
    include/${MODULE_NAME}/MCLabel.h
    include/${MODULE_NAME}/HitType.h
    )

Set(LINKDEF src/DataFormatsFITT0LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()


