# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHCalibration
         SOURCES
           src/PedestalDigit.cxx
           src/PedestalProcessor.cxx
           src/PedestalCalibrator.cxx
           PUBLIC_LINK_LIBRARIES Microsoft.GSL::GSL O2::DetectorsCalibration O2::CCDB O2::MCHBase O2::MCHRawDecoder O2::DataFormatsMCH)

o2_target_root_dictionary(MCHCalibration
  HEADERS include/MCHCalibration/PedestalDigit.h include/MCHCalibration/PedestalProcessor.h include/MCHCalibration/PedestalCalibrator.h)

o2_add_executable(pedestal-decoding-workflow
        SOURCES src/pedestal-decoding-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES Boost::program_options O2::Framework O2::DPLUtils O2::MCHCalibration)

o2_add_executable(mch-pedestal-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/pedestal-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::MCHCalibration O2::DetectorsCalibration)


