# @author Gvozden Neskovic
# @brief  DataDistribution common stuff: DataModel and Utilities

o2_define_bucket(
    NAME
    DataDistributionCommon_bucket

    DEPENDENCIES
    DataDistributionBase_bucket
    root_base_bucket Core Gpad Hist
)

set(MODULE_NAME "DataDistributionCommon")
set(MODULE_BUCKET_NAME DataDistributionCommon_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/ReadoutDataModel.cxx
    src/SubTimeFrameDataModel.cxx
    src/SubTimeFrameVisitors.cxx
    src/SubTimeFrameUtils.cxx
    src/FilePathUtils.cxx
    src/SubTimeFrameFile.cxx
    src/SubTimeFrameFileWriter.cxx
    src/SubTimeFrameFileSink.cxx
    src/SubTimeFrameFileReader.cxx
    src/SubTimeFrameFileSource.cxx
    src/SubTimeFrameDPL.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/test_FilePathUtils.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
