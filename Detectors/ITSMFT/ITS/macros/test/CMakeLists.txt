# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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

o2_add_test_root_macro(CheckLookUp.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                       LABELS its)

o2_add_test_root_macro(CheckLUtime.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                       LABELS its)

o2_add_test_root_macro(CheckTopologies.C
                       PUBLIC_LINK_LIBRARIES O2::MathUtils
                                             O2::ITSBase
                                             O2::ITSMFTReconstruction
                                             O2::ITSMFTSimulation
                                             O2::DataFormatsITSMFT
                                             O2::SimulationDataFormat
                       LABELS its)

o2_add_test_root_macro(CheckTracks.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::DataFormatsITS
                                             O2::DataFormatsITSMFT
                       LABELS its)

o2_add_test_root_macro(DisplayTrack.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTSimulation
                                             O2::DataFormatsITS
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                       LABELS its)

o2_add_test_root_macro(dictionary_integrity_test.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT
                                             O2::ITSMFTReconstruction
                       LABELS its)

o2_add_test_root_macro(run_buildTopoDict_its.C
                       PUBLIC_LINK_LIBRARIES O2::MathUtils
                                             O2::ITSBase
                                             O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTSimulation
                                             O2::SimulationDataFormat
                       LABELS its)
