set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Cluster.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Cluster.h
    )

Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)

O2_GENERATE_LIBRARY()

