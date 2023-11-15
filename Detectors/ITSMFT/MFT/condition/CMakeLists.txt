# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MFTCondition
               SOURCES src/MFTDCSProcessor.cxx
                       src/DCSNameResolver.cxx
                       src/DCSConfigInfo.cxx
                       src/DCSConfigReader.cxx
                       src/DCSConfigUtils.cxx
               PUBLIC_LINK_LIBRARIES O2::CCDB
                                     O2::DataFormatsMFT
                                     O2::DetectorsCalibration
                                     O2::DetectorsDCS
                                     Microsoft.GSL::GSL)
o2_target_root_dictionary(MFTCondition
                          HEADERS include/MFTCondition/MFTDCSProcessor.h
                                  include/MFTCondition/DCSNameResolver.h
                                  include/MFTCondition/DCSConfigInfo.h
                                  include/MFTCondition/DCSConfigReader.h
                                  include/MFTCondition/DCSConfigUtils.h
                          LINKDEF src/MFTConditionLinkDef.h)

o2_add_executable(mft-dcs-sim-workflow
                  COMPONENT_NAME condition
                  SOURCES testWorkflow/mft-dcs-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(mft-dcs-workflow
                  COMPONENT_NAME condition
                  SOURCES testWorkflow/mft-dcs-data-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::MFTCondition
                                        O2::DetectorsDCS)

o2_add_executable(mft-dcs-config-workflow
                  COMPONENT_NAME condition
                  SOURCES testWorkflow/mft-dcs-config-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::DetectorsCommonDataFormats
                                        O2::MFTCondition
                                        O2::DetectorsDCS)

add_subdirectory(macros)
