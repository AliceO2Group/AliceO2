set(MODULE_NAME "TPCbase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/ContainerFactory.cxx
   src/CRU.cxx
   src/DigitPos.cxx
   src/FECInfo.cxx
   src/Mapper.cxx
   src/PadInfo.cxx
   src/PadPos.cxx
   src/PadRegionInfo.cxx
   src/PadSecPos.cxx
   src/Point2D.cxx
   src/Point3D.cxx
   src/ROC.cxx
   src/Sector.cxx
)

set(HEADERS
   include/ContainerFactory.h
   include/CRU.h
#    include/Defs.h
   include/DigitPos.h
   include/FECInfo.h
   include/Mapper.h
   include/PadInfo.h
   include/PadPos.h
   include/PadRegionInfo.h
   include/PadSecPos.h
   include/Point2D.h
   include/Point3D.h
   include/ROC.h
   include/Defs.h
   include/Sector.h
)

Set(LINKDEF src/tpcBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_base_bucket)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
