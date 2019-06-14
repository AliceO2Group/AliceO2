set(MODULE_NAME "MIDBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ChamberEfficiency.cxx
  src/Constants.cxx
  src/GeometryTransformer.cxx
  src/Mapping.cxx
  src/MpArea.cxx
)

set(NO_DICT_HEADERS
  include/${MODULE_NAME}/ChamberEfficiency.h
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/GeometryTransformer.h
  include/${MODULE_NAME}/Mapping.h
  include/${MODULE_NAME}/MpArea.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_base_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)
