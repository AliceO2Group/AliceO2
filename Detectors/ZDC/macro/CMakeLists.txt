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

o2_add_test_root_macro(CreateSimCondition.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCSimulation
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateSimCondition_pp.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCSimulation
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)


o2_add_test_root_macro(CreateModuleConfig.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(readZDCDigits.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCSimulation O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateRecoConfigZDC.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateTDCCalib.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateTDCCalibConfig.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(CreateEnergyCalib.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateTowerCalib.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateTDCCorr.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                       LABELS zdc)

o2_add_test_root_macro(CreateInterCalibConfig.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(CreateWaveformCalibConfig.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(CreateWaveformCalibParam.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(CreateBaselineCalibConfig.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(CreateBaselineCalib.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)

o2_add_test_root_macro(InspectCCDBFile.C
                       PUBLIC_LINK_LIBRARIES O2::ZDCBase O2::ZDCReconstruction
                                             O2::SimulationDataFormat O2::CCDB
                                             O2::ZDCCalib
                       LABELS zdc)
