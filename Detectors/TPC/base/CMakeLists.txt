set(MODULE_NAME "TPCbase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/ContainerFactory.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/ContainerFactory.h
    )


Set(LINKDEF src/tpcBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_base_bucket)

O2_GENERATE_LIBRARY()


