# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHRawDecoder
               SOURCES src/Decoder.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHRawCommon O2::MCHRawElecMap
               PRIVATE_LINK_LIBRARIES O2::MCHRawImplHelpers)

if(BUILD_TESTING)
  o2_add_test(decoder
              SOURCES src/testDecoder.cxx
              COMPONENT_NAME mch
              LABELS "mch;raw"
              PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHTestRawRefBuffers)

  o2_add_test(bare-elink-decoder
              SOURCES src/testBareElinkDecoder.cxx
              COMPONENT_NAME mch
              LABELS "mch;raw"
              PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHRawImplHelpers)

  o2_add_test(userlogic-elink-decoder
          SOURCES src/testUserLogicElinkDecoder.cxx
              COMPONENT_NAME mch
              LABELS "mch;raw"
              PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHRawImplHelpers)

endif()
