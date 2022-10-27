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

o2_add_library(MCHCalibration
         SOURCES
           src/BadChannelCalibrationDevice.cxx
           src/BadChannelCalibrator.cxx
           src/BadChannelCalibratorParam.cxx
           src/PedestalChannel.cxx
           src/PedestalData.cxx
           src/PedestalDigit.cxx
           PUBLIC_LINK_LIBRARIES Microsoft.GSL::GSL O2::DetectorsCalibration O2::CCDB O2::MCHBase O2::MCHRawDecoder O2::DataFormatsMCH)

o2_target_root_dictionary(MCHCalibration HEADERS include/MCHCalibration/BadChannelCalibrator.h
                                                 include/MCHCalibration/BadChannelCalibratorParam.h
                                                 include/MCHCalibration/PedestalDigit.h)

o2_add_executable(pedestal-decoding-workflow
        SOURCES src/pedestal-decoding-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES Boost::program_options O2::Framework O2::DPLUtils O2::MCHCalibration O2::MCHConstants)

o2_add_executable(mch-badchannel-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/badchannel-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::MCHCalibration O2::DetectorsCalibration)

if(BUILD_TESTING)

o2_add_test(pedestal-data-iterator
            SOURCES test/testPedestalData.cxx
            COMPONENT_NAME mch
            LABELS "muon;mch"
            PUBLIC_LINK_LIBRARIES O2::MCHCalibration Boost::boost)

endif()
