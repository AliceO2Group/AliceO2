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

o2_add_test_root_macro(CheckClusterShape.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase O2::ITSMFTSimulation
                                             O2::SimulationDataFormat
                       LABELS its)

o2_add_test_root_macro(CheckClusters.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase O2::DataFormatsITSMFT
                                             O2::ITSMFTSimulation O2::MathUtils
                                             O2::SimulationDataFormat
                       LABELS its)

o2_add_test_root_macro(CheckDigits.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase
                                             O2::ITSMFTBase
                                             O2::ITSMFTSimulation
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                                             O2::DetectorsBase
                       LABELS its)

#o2_add_test_root_macro(CheckLUtime.C
#                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
#                                             O2::DataFormatsITSMFT
#                       LABELS its)

o2_add_test_root_macro(CreateDictionaries.C
                       PUBLIC_LINK_LIBRARIES O2::MathUtils
                                             O2::ITSBase
                                             O2::ITSMFTReconstruction
                                             O2::ITSMFTSimulation
                                             O2::DataFormatsITSMFT
                                             O2::SimulationDataFormat
                       LABELS its)

#o2_add_test_root_macro(CheckCOG.C
#                       PUBLIC_LINK_LIBRARIES O2::MathUtils
#                                             O2::ITSBase
#                                             O2::ITSMFTReconstruction
#                                             O2::ITSMFTSimulation
#                                             O2::DataFormatsITSMFT
#                                             O2::SimulationDataFormat
#                       LABELS its)

o2_add_test_root_macro(CheckTracks.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::ITSBase
                                             O2::DataFormatsITS
                                             O2::DataFormatsITSMFT
                       LABELS its)

o2_add_test_root_macro(CheckTracksCA.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::ITSBase
                                             O2::DataFormatsITS
                                             O2::DataFormatsITSMFT
                       LABELS its)

o2_add_test_root_macro(RunGPUTracking.C
                       PUBLIC_LINK_LIBRARIES O2::GPUTracking
                       LABELS its)

o2_add_test_root_macro(DisplayTrack.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTSimulation
                                             O2::DataFormatsITS
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                       LABELS its)

#o2_add_test_root_macro(dictionary_integrity_test.C
#                       PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT
#                                             O2::ITSMFTReconstruction
#                       LABELS its)

o2_add_test_root_macro(run_digi2rawVarPage_its.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTBase
                                             O2::ITSMFTSimulation
                       LABELS its)

o2_add_test_root_macro(ITSMisaligner.C
                       PUBLIC_LINK_LIBRARIES O2::CCDB
                                             O2::ITSReconstruction
                       LABELS its)
