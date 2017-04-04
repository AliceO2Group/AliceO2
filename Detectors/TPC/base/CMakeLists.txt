set(MODULE_NAME "TPCBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/CalArray.cxx
   src/CalDet.cxx
   src/ContainerFactory.cxx
   src/CRU.cxx
   src/Digit.cxx
   src/DigitPos.cxx
   src/FECInfo.cxx
   src/Mapper.cxx
   src/PadInfo.cxx
   src/PadPos.cxx
   src/PadRegionInfo.cxx
   src/PadSecPos.cxx
   src/Painter.cxx
   src/PartitionInfo.cxx
   src/Point2D.cxx
   src/Point3D.cxx
   src/RandomRing.cxx
   src/ROC.cxx
   src/Sector.cxx
)

set(HEADERS
   include/TPCBase/CalArray.h
   include/TPCBase/CalDet.h
   include/TPCBase/ContainerFactory.h
   include/TPCBase/CRU.h
#    include/TPCBase/Defs.h
   include/TPCBase/Digit.h
   include/TPCBase/DigitPos.h
   include/TPCBase/FECInfo.h
   include/TPCBase/Mapper.h
   include/TPCBase/PadInfo.h
   include/TPCBase/PadPos.h
   include/TPCBase/PadRegionInfo.h
   include/TPCBase/PadSecPos.h
   include/TPCBase/Painter.h
   include/TPCBase/PartitionInfo.h
   include/TPCBase/Point2D.h
   include/TPCBase/Point3D.h
   include/TPCBase/RandomRing.h
   include/TPCBase/ROC.h
   include/TPCBase/Defs.h
   include/TPCBase/Sector.h
)

Set(LINKDEF src/TPCBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_base_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
   test/testTPCBase.cxx
   test/testTPCCalDet.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)

install(
  DIRECTORY files
  DESTINATION share/Detectors/TPC/
)
