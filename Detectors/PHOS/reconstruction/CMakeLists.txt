SET(MODULE_NAME PHOSReconstruction)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Cluster.cxx
  src/Clusterer.cxx
  src/ClustererTask.cxx
 )

set(HEADERS
  include/PHOSReconstruction/Cluster.h
  include/PHOSReconstruction/Clusterer.h
  include/PHOSReconstruction/ClustererTask.h
)

SET(LINKDEF src/PHOSReconstructionLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME phos_reconstruction_bucket)

O2_GENERATE_LIBRARY()
