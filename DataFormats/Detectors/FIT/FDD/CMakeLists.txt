set(MODULE_NAME "DataFormatsFDD")
set(MODULE_BUCKET_NAME data_format_fit_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    )

set(HEADERS
    include/${MODULE_NAME}/Digit.h
    include/${MODULE_NAME}/Hit.h
    include/${MODULE_NAME}/MCLabel.h
    )

Set(LINKDEF src/DataFormatsFDDLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()


