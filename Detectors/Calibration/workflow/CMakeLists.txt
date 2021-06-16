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

o2_add_library(DetectorsCalibrationWorkflow
               SOURCES src/MeanVertexCalibratorSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                     O2::Headers
                     O2::CCDB
                     O2::CommonUtils
                     Microsoft.GSL::GSL
                     O2::ReconstructionDataFormats
                     O2::DataFormatsCalibration
                     O2::DetectorsCalibration)

o2_add_executable(mean-vertex-calibration-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/mean-vertex-calibration-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                    O2::DetectorsCalibration
                    O2::DetectorsCalibrationWorkflow
                    O2::ReconstructionDataFormats
                    O2::DataFormatsCalibration
                    O2::CCDB)
