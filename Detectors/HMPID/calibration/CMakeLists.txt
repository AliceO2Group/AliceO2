#Copyright 2019 - 2020 CERN and copyright holders of ALICE O2.
#See https: // alice-o2.web.cern.ch/copyright for details of the copyright holders.
#All rights not expressly granted are reserved.
#
#This software is distributed under the terms of the GNU General Public
#License v3(GPL Version 3), copied verbatim in the file "COPYING".
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(HMPIDCalibration
               TARGETVARNAME targetName
               SOURCES src/HMPIDDCSProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::HMPIDBase
                                     O2::Framework
                                     O2::DataFormatsHMP
                                     O2::HMPIDSimulation
                                     O2::SimulationDataFormat
                                     O2::DetectorsBase
                                     O2::DetectorsCalibration
                                     O2::DetectorsDCS
                                     O2::DetectorsRaw
                                     O2::CCDB
                                     ROOT::Physics
                                     ROOT::Minuit
                                     O2::Steer
                                     Microsoft.GSL::GSL)

o2_target_root_dictionary(HMPIDCalibration
                          HEADERS  include/HMPIDCalibration/HMPIDDCSProcessor.h)


if (OpenMP_CXX_FOUND)
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_add_executable(hmpid-dcs-sim-workflow# run with o2-calibration-hmpid-dcs-workflow
                  COMPONENT_NAME calibration # adds calibration to the above command
                  SOURCES testWorkflow/hmpid-dcs-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::HMPIDCalibration
                                        O2::DCStestWorkflow)

o2_add_executable(hmpid-dcs-workflow # run with o2-calibration-hmpid-dcs-workflow
                  COMPONENT_NAME calibration # adds calibration to the above command
                  SOURCES testWorkflow/hmpid-dcs-data-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::HMPIDCalibration
                                        O2::DCStestWorkflow)
