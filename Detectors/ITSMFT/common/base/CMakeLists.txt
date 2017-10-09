set(MODULE_NAME "ITSMFTBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
     src/SDigit.cxx
     src/Digit.cxx
     src/SegmentationAlpide.cxx
#    src/ContainerFactory.cxx
#    src/MisalignmentParameter.cxx
     src/GeometryTGeo.cxx
    )

set(HEADERS
     include/${MODULE_NAME}/SDigit.h
     include/${MODULE_NAME}/Digit.h
     include/${MODULE_NAME}/SegmentationAlpide.h
#    include/${MODULE_NAME}/ContainerFactory.h
#    include/${MODULE_NAME}/MisalignmentParameter.h
     include/${MODULE_NAME}/GeometryTGeo.h
    )


Set(LINKDEF src/ITSMFTBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME itsmft_base_bucket)

O2_GENERATE_LIBRARY()
