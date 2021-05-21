# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsCalibration
               SOURCES src/TimeSlot.cxx
                   src/TimeSlotCalibration.cxx
                   src/Utils.cxx
                   src/MeanVertexData.cxx
                   src/MeanVertexCalibrator.cxx
                   src/MeanVertexParams.cxx
               PUBLIC_LINK_LIBRARIES O2::Headers
                   O2::CCDB
                   O2::CommonUtils
                   ROOT::Minuit
                   Microsoft.GSL::GSL
                   O2::ReconstructionDataFormats
                   O2::SimConfig
                   O2::DataFormatsCalibration)

o2_target_root_dictionary(DetectorsCalibration
                          HEADERS include/DetectorsCalibration/TimeSlotCalibration.h
                                  include/DetectorsCalibration/TimeSlot.h
                                  include/DetectorsCalibration/Utils.h
                                  include/DetectorsCalibration/MeanVertexData.h
                                  include/DetectorsCalibration/MeanVertexCalibrator.h
                                  include/DetectorsCalibration/MeanVertexParams.h)

o2_add_executable(ccdb-populator-workflow
                  COMPONENT_NAME calibration
                  SOURCES workflow/ccdb-populator-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                      O2::TOFCalibration
                      O2::DetectorsCalibration
                      O2::DataFormatsTOF
                      O2::CCDB)

add_subdirectory(workflow)
add_subdirectory(testMacros)
