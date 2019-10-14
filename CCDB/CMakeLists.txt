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
               SOURCES  src/CcdbApi.cxx
                        src/BasicCCDBManager.cxx
                        src/CCDBTimeStampUtils.cxx
        src/IdPath.cxx
        PUBLIC_LINK_LIBRARIES CURL::libcurl
                                    FairRoot::ParMQ
                                    ROOT::Hist
                                    O2::CommonUtils
                                    FairMQ::FairMQ
               TARGETVARNAME targetName)

o2_target_root_dictionary(CCDB
                          HEADERS
                                  include/CCDB/CcdbApi.h
        include/CCDB/TObjectWrapper.h
        include/CCDB/IdPath.h
        include/CCDB/BasicCCDBManager.h
                                  include/CCDB/CCDBTimeStampUtils.h)

o2_add_test(CcdbApi
            SOURCES test/testCcdbApi.cxx
            COMPONENT_NAME ccdb
            PUBLIC_LINK_LIBRARIES O2::CCDB
            LABELS ccdb)
