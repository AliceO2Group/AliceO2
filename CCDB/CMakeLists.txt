# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CCDB
               SOURCES src/Backend.cxx
                       src/BackendOCDB.cxx
                       src/BackendRiak.cxx
                       src/Condition.cxx
                       src/ConditionId.cxx
                       src/ConditionMetaData.cxx
                       src/FileStorage.cxx
                       src/GridStorage.cxx
                       src/IdPath.cxx
                       src/IdRunRange.cxx
                       src/LocalStorage.cxx
                       src/Manager.cxx
                       src/ObjectHandler.cxx
                       src/Storage.cxx
                       src/XmlHandler.cxx
                       src/CcdbApi.cxx
                       src/ConditionsMQServer.cxx
                       src/ConditionsMQClient.cxx
                       src/BasicCCDBManager.cxx
                       src/CCDBTimeStampUtils.cxx
                       src/request.proto
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ
                                     protobuf::libprotobuf
                                     ROOT::Hist
                                     ROOT::XMLParser
                                     O2::CommonUtils
                                     FairRoot::ParMQ
                                     O2::Device
                                     CURL::libcurl
                                     Boost::thread
               TARGETVARNAME targetName)

protobuf_generate(TARGET ${targetName})

o2_target_root_dictionary(CCDB
                          HEADERS include/CCDB/Backend.h
                                  include/CCDB/BackendOCDB.h
                                  include/CCDB/BackendRiak.h
                                  include/CCDB/Condition.h
                                  include/CCDB/ConditionId.h
                                  include/CCDB/ConditionMetaData.h
                                  include/CCDB/FileStorage.h
                                  include/CCDB/GridStorage.h
                                  include/CCDB/IdPath.h
                                  include/CCDB/IdRunRange.h
                                  include/CCDB/LocalStorage.h
                                  include/CCDB/Manager.h
                                  include/CCDB/ObjectHandler.h
                                  include/CCDB/Storage.h
                                  include/CCDB/XmlHandler.h
                                  include/CCDB/TObjectWrapper.h
                                  include/CCDB/CcdbApi.h
                                  include/CCDB/BasicCCDBManager.h
                                  include/CCDB/CCDBTimeStampUtils.h
                                  test/TestClass.h)

o2_add_executable(conditions-server
                  SOURCES src/runConditionsServer.cxx
                  COMPONENT_NAME ccdb
                  PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_add_executable(conditions-client
                  SOURCES src/runConditionsClient.cxx
                  COMPONENT_NAME ccdb
                  PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_add_executable(standalone-client
                  SOURCES test/testQueryServerStandalone.cxx
                  COMPONENT_NAME ccdb
                  PUBLIC_LINK_LIBRARIES O2::CCDB)

o2_data_file(COPY config/conditions-server.json config/conditions-client.json
                  example/fill_local_ocdb.C
             DESTINATION config)

o2_add_test(WriteReadAny
            SOURCES test/testWriteReadAny.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

o2_add_test(CcdbApi
            SOURCES test/testCcdbApi.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)

