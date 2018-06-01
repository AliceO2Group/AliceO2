set(MODULE_NAME "FITReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/RecPoints.cxx
  src/CollisionTimeRecoTask.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/RecPoints.h
    include/${MODULE_NAME}/CollisionTimeRecoTask.h
    )


Set(LINKDEF src/FITReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_reconstruction_bucket)

O2_GENERATE_LIBRARY()


