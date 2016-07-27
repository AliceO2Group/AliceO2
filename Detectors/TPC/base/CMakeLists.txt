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
   include/TPCbase/ContainerFactory.h
   include/TPCbase/CRU.h
#    include/TPCbase/Defs.h
   include/TPCbase/DigitPos.h
   include/TPCbase/FECInfo.h
   include/TPCbase/Mapper.h
   include/TPCbase/PadInfo.h
   include/TPCbase/PadPos.h
   include/TPCbase/PadRegionInfo.h
   include/TPCbase/PadSecPos.h
   include/TPCbase/Point2D.h
   include/TPCbase/Point3D.h
   include/TPCbase/ROC.h
   include/TPCbase/Defs.h
   include/TPCbase/Sector.h
)

Set(LINKDEF src/tpcBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_base_bucket)

O2_GENERATE_LIBRARY()
