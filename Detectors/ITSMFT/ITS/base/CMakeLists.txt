set(MODULE_NAME "ITSBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/UpgradeGeometryTGeo.cxx
    src/GeometryManager.cxx
    src/Segmentation.cxx
    src/UpgradeSegmentationPixel.cxx
    src/ContainerFactory.cxx
    src/MisalignmentParameter.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/UpgradeGeometryTGeo.h
    include/${MODULE_NAME}/GeometryManager.h
    include/${MODULE_NAME}/Segmentation.h
    include/${MODULE_NAME}/UpgradeSegmentationPixel.h
    include/${MODULE_NAME}/ContainerFactory.h
    include/${MODULE_NAME}/MisalignmentParameter.h
    )


Set(LINKDEF src/ITSBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME its_base_bucket)

O2_GENERATE_LIBRARY()
