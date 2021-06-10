# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHRawEncoderDigit
        SOURCES
        Digit2ElecMapper.cxx
        DigitPayloadEncoder.cxx
        DigitRawEncoder.cxx
        DigitTreeReader.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DataFormatsParameters
        O2::MCHRawEncoderPayload)

o2_add_executable(digits-to-json
        SOURCES digits-to-json.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
        Boost::program_options
        O2::MCHMappingImpl4
        O2::MCHRawElecMap
        O2::MCHRawEncoderDigit
        O2::MCHSimulation
        O2::Steer
        fmt::fmt)

o2_add_executable(digits-to-raw
        SOURCES digits-to-raw.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
        Boost::program_options
        O2::DetectorsRaw
        O2::MCHMappingImpl4
        O2::MCHRawElecMap
        O2::MCHRawEncoderDigit
        O2::MCHRawImplHelpers
        O2::MCHSimulation
        O2::Steer
        fmt::fmt)

o2_add_test(digit-tree-reader
            SOURCES testDigitTreeReader.cxx
            COMPONENT_NAME mch
            LABELS muon mch
            PUBLIC_LINK_LIBRARIES O2::MCHRawEncoderDigit)
