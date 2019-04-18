set(MODULE_NAME "MIDBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ChamberEfficiency.cxx
  src/Constants.cxx
  src/GeometryTransformer.cxx
  src/HitFinder.cxx
  src/Mapping.cxx
  src/MpArea.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_base_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)
