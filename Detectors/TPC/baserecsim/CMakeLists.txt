# Copyright 2019-2025 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(TPCBaseRecSim
               SOURCES src/DeadChannelMapCreator.cxx
                       src/Painter.cxx
                       src/CDBInterface.cxx
               PUBLIC_LINK_LIBRARIES O2::TPCBase)

o2_target_root_dictionary(TPCBaseRecSim
                          EXTRA_PATCH src/TPCFlagsMemberCustomStreamer.cxx
                          HEADERS include/TPCBaseRecSim/Painter.h
                                  include/TPCBaseRecSim/PadFlags.h
                                  include/TPCBaseRecSim/DeadChannelMapCreator.h
                                  include/TPCBaseRecSim/CDBTypes.h
                                  include/TPCBaseRecSim/CDBInterface.h)

if(BUILD_SIMULATION)
  # this test needs CCDB/XROOTD which is for sure
  # available in the default-o2 software stack
  o2_add_test(CalDet
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBaseRecSim
            SOURCES test/testTPCCalDet.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc)
endif()
