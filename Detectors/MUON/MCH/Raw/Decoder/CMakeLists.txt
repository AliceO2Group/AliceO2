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
        SOURCES src/BareELinkDecoder.cxx
                src/DataDecoder.cxx
                src/OrbitInfo.cxx
                src/PageDecoder.cxx
                src/ROFFinder.cxx
                src/UserLogicElinkDecoder.cxx
                src/RDHManip.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
                              O2::MCHBase
                              O2::MCHMappingImpl4
                              O2::MCHMappingInterface
                              O2::MCHRawCommon
                              O2::MCHRawElecMap
                              O2::DataFormatsMCH
        PRIVATE_LINK_LIBRARIES O2::MCHRawImplHelpers)

if(BUILD_TESTING)

        o2_add_test(bare-elink-decoder
                SOURCES src/testBareElinkDecoder.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHRawImplHelpers)

        o2_add_test(userlogic-endpoint-decoder
                SOURCES src/testUserLogicEndpointDecoder.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHRawImplHelpers
                O2::MCHRawEncoderPayload)

        o2_add_test(rdh-manip
                SOURCES src/testRDHManip.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder Boost::boost)

        o2_add_test(digits-time-computation
                SOURCES src/testDigitsTimeComputation.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder Boost::boost)

        o2_add_test(rof-finder
                SOURCES src/testROFFinder.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder Boost::boost)

endif()
