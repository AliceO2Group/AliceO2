# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(CCDB
               SOURCES  src/CcdbApi.cxx
                        src/CCDBDownloader.cxx
                        src/BasicCCDBManager.cxx
                        src/CCDBTimeStampUtils.cxx
        src/IdPath.cxx src/CCDBQuery.cxx
        PUBLIC_LINK_LIBRARIES CURL::libcurl
                                    ROOT::Hist
                                    O2::CommonUtils
                                    FairMQ::FairMQ
                                    libjalien::libjalienO2
                                    LibUV::LibUV
               TARGETVARNAME targetName)

o2_target_root_dictionary(CCDB
                          HEADERS
                                  include/CCDB/CcdbApi.h
                                  include/CCDB/CcdbObjectInfo.h
                                  include/CCDB/TObjectWrapper.h
                                  include/CCDB/IdPath.h
                                  include/CCDB/BasicCCDBManager.h
                                  include/CCDB/CCDBTimeStampUtils.h
                                  include/CCDB/CCDBQuery.h
                                  include/CCDB/CCDBDownloader.h)

o2_add_executable(inspectccdbfile
            COMPONENT_NAME ccdb
            SOURCES src/InspectCCDBFile.cxx
            PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_add_executable(upload
            COMPONENT_NAME ccdb
            SOURCES src/UploadTool.cxx
            PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_add_executable(downloadccdbfile
            COMPONENT_NAME ccdb
            SOURCES src/DownloadCCDBFile.cxx
            PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_add_test(CcdbApi
            SOURCES test/testCcdbApi.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

o2_add_test(CcdbApiConfigParam
    SOURCES test/testCcdbApi_ConfigParam.cxx
    COMPONENT_NAME ccdb
    PUBLIC_LINK_LIBRARIES O2::CCDB O2::DetectorsVertexing
    LABELS ccdb)


o2_add_test(CcdbApi-Alien
    SOURCES test/testCcdbApi_alien.cxx
    COMPONENT_NAME ccdb
    PUBLIC_LINK_LIBRARIES O2::CCDB
    LABELS ccdb)

o2_add_test(BasicCCDBManager
            SOURCES test/testBasicCCDBManager.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

o2_add_test(CcdbApiMultipleUrls
            SOURCES test/testCcdbApiMultipleUrls.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

o2_add_test(CcdbDownloader
            SOURCES test/testCcdbApiDownloader.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

# extra CcdbApi test which dispatches to CCDBDownloader (tmp until full move done)
o2_add_test_command(NAME CcdbApi-MultiHandle
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    COMMAND ${CMAKE_BINARY_DIR}/stage/tests/o2-test-ccdb-CcdbApi
                    ENVIRONMENT "ALICEO2_ENABLE_MULTIHANDLE_CCDBAPI=1")
