set(MODULE_NAME "HMPIDBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Param.cxx
    src/Digit.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Param.h
    include/${MODULE_NAME}/Digit.h
    )

Set(LINKDEF src/HMPIDBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME hmpid_base_bucket)

O2_GENERATE_LIBRARY()
