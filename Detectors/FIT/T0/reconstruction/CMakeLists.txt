set(MODULE_NAME "T0Reconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/CollisionTimeRecoTask.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/CollisionTimeRecoTask.h
    )


Set(LINKDEF src/T0ReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_reconstruction_bucket)

O2_GENERATE_LIBRARY()


