set(MODULE_NAME "HitAnalysis")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/HitAnalysis.cxx
    )
set(HEADERS
    include/HitAnalysis/HitAnalysis.h
    )
Set(LINKDEF src/HitAnalysisLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME hitanalysis_bucket)


O2_GENERATE_LIBRARY()


