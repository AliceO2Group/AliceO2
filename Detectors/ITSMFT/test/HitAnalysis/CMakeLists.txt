set(MODULE_NAME "HitAnalysis")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/HitAnalysis.cxx
    )
set(HEADERS
    include/HitAnalysis/HitAnalysis.h
    )
Set(LINKDEF src/testitsLinkDef.h)
Set(LIBRARY_NAME testits)
set(BUCKET_NAME itsmft_test)


O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
