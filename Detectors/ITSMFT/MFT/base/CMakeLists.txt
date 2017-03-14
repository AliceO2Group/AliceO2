set(MODULE_NAME "MFTBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Constants.cxx
  src/GeometryTGeo.cxx
  src/Geometry.cxx
  src/GeometryBuilder.cxx
  src/VSegmentation.cxx
  src/Segmentation.cxx
  src/HalfSegmentation.cxx
  src/HalfDiskSegmentation.cxx
  src/LadderSegmentation.cxx
  src/ChipSegmentation.cxx
  src/HalfDetector.cxx
  src/HalfDisk.cxx
  src/Plane.cxx
  src/Ladder.cxx
  src/Flex.cxx
  src/Chip.cxx
  src/Support.cxx
  src/HeatExchanger.cxx
  src/HalfCone.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/GeometryTGeo.h
  include/${MODULE_NAME}/Geometry.h
  include/${MODULE_NAME}/GeometryBuilder.h
  include/${MODULE_NAME}/VSegmentation.h
  include/${MODULE_NAME}/Segmentation.h
  include/${MODULE_NAME}/HalfSegmentation.h
  include/${MODULE_NAME}/HalfDiskSegmentation.h
  include/${MODULE_NAME}/LadderSegmentation.h
  include/${MODULE_NAME}/ChipSegmentation.h
  include/${MODULE_NAME}/HalfDetector.h
  include/${MODULE_NAME}/HalfDisk.h
  include/${MODULE_NAME}/Plane.h
  include/${MODULE_NAME}/Ladder.h
  include/${MODULE_NAME}/Flex.h
  include/${MODULE_NAME}/Chip.h
  include/${MODULE_NAME}/Support.h
  include/${MODULE_NAME}/HeatExchanger.h
  include/${MODULE_NAME}/HalfCone.h
)

Set(LINKDEF src/MFTBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mft_base_bucket)

O2_GENERATE_LIBRARY()

