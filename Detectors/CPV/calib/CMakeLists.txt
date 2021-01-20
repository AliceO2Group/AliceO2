# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CPVCalib
               SOURCES src/BadChannelMap.cxx
                       src/CalibParams.cxx
                       src/Pedestals.cxx
	       PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase)

o2_target_root_dictionary(CPVCalib
                          HEADERS include/CPVCalib/BadChannelMap.h
                                  include/CPVCalib/CalibParams.h
                                  include/CPVCalib/Pedestals.h
                          LINKDEF src/CPVCalibLinkDef.h)

add_subdirectory(CPVCalibWorkflow)

if(BUILD_TESTING)
  o2_add_test_root_macro(macros/PostBadMapCCDB.C  
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::CPVCalib
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro( macros/PostCalibCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::CPVCalib
                         LABELS CPV COMPILE_ONLY)

endif()

