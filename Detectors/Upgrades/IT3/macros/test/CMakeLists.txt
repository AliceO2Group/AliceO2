# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test_root_macro(CheckDigitsITS3.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase
                                             O2::ITS3Base
                                             O2::ITSMFTBase
                                             O2::ITSMFTSimulation
                                             O2::ITS3Simulation
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                                             O2::DetectorsBase
                       LABELS its3)

o2_add_test_root_macro(CheckClustersITS3.C
                       PUBLIC_LINK_LIBRARIES O2::ITSBase
                                             O2::ITS3Base
                                             O2::ITSMFTBase
                                             O2::ITSMFTSimulation
                                             O2::ITS3Simulation
                                             O2::ITS3Reconstruction
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                                             O2::DetectorsBase
                       LABELS its3)