o2_add_library(CommonUtils
               SOURCES
               src/TreeStream.cxx
               src/TreeStreamRedirector.cxx
               src/RootChain.cxx
               src/CompStream.cxx
               src/ShmManager.cxx
               PUBLIC_LINK_LIBRARIES
               ROOT::Tree
               Boost::iostreams
               FairLogger::FairLogger)

o2_target_root_dictionary(CommonUtils
                          HEADERS
                          include/CommonUtils/TreeStream.h
                          include/CommonUtils/TreeStreamRedirector.h
                          include/CommonUtils/RootChain.h
                          include/CommonUtils/BoostSerializer.h
                          include/CommonUtils/ShmManager.h
                          include/CommonUtils/RngHelper.h
                          include/CommonUtils/StringUtils.h)

# set(TEST_SRCS test/testTreeStream.cxx test/testBoostSerializer.cxx
# test/testCompStream.cxx )
#
# O2_GENERATE_TESTS( MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${BUCKET_NAME} TEST_SRCS ${TEST_SRCS} )
