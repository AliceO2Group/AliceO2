set(MODULE_NAME "DetectorsPassive")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Absorber.cxx
  src/Cave.cxx
  src/Dipole.cxx
  src/Pipe.cxx
  src/Magnet.cxx
  src/PassiveContFact.cxx
  src/FrameStructure.cxx
  src/Shil.cxx
  src/Hall.cxx
)

Set(HEADERS
    include/${MODULE_NAME}/Absorber.h
    include/${MODULE_NAME}/Cave.h
    include/${MODULE_NAME}/Dipole.h
    include/${MODULE_NAME}/Magnet.h
    include/${MODULE_NAME}/PassiveContFact.h
    include/${MODULE_NAME}/Pipe.h
    include/${MODULE_NAME}/FrameStructure.h
    include/${MODULE_NAME}/Shil.h
    include/${MODULE_NAME}/Hall.h
)

Set(LINKDEF src/PassiveLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME passive_detector_bucket)

O2_GENERATE_LIBRARY()
