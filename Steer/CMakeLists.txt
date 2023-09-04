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

o2_add_library(Steer
               SOURCES src/O2MCApplication.cxx
                       src/HitProcessingManager.cxx src/MCKinematicsReader.cxx
                       src/MaterialBudgetMap.cxx src/O2MCApplicationEvalMat.cxx
                       PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                             O2::CommonConstants
                                             O2::DetectorsBase
                                             O2::SimulationDataFormat
                                             O2::DetectorsCommonDataFormats
                                             O2::Generators O2::SimConfig)

o2_add_executable(colcontexttool
                  COMPONENT_NAME steer
                  SOURCES src/CollisionContextTool.cxx
                  PUBLIC_LINK_LIBRARIES Boost::program_options O2::Algorithm O2::Steer O2::SimulationDataFormat)

o2_target_root_dictionary(Steer
                          HEADERS include/Steer/HitProcessingManager.h
                                  include/Steer/O2RunSim.h
                                  include/Steer/O2MCApplication.h
                                  include/Steer/O2MCApplicationBase.h
                                  include/Steer/O2MCApplicationEvalMat.h
                                  include/Steer/MCKinematicsReader.h
                                  include/Steer/MaterialBudgetMap.h)
o2_add_test(InteractionSampler
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
            SOURCES test/testInteractionSampler.cxx
            LABELS steer)

o2_add_test(HitProcessingManager
            PUBLIC_LINK_LIBRARIES O2::Steer
            SOURCES test/testHitProcessingManager.cxx
            LABELS steer)

add_subdirectory(DigitizerWorkflow)
