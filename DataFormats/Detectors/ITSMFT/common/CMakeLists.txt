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

o2_add_library(DataFormatsITSMFT
               SOURCES src/ROFRecord.cxx
                       src/Digit.cxx
                       src/NoiseMap.cxx
                       src/Cluster.cxx
                       src/CompCluster.cxx
                       src/ClusterPattern.cxx
                       src/ClusterTopology.cxx
                       src/TopologyDictionary.cxx
                       src/CTF.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                       O2::ReconstructionDataFormats
                       Microsoft.GSL::GSL)

o2_target_root_dictionary(DataFormatsITSMFT
                          HEADERS include/DataFormatsITSMFT/ROFRecord.h
                                 include/DataFormatsITSMFT/Digit.h
                                 include/DataFormatsITSMFT/GBTCalibData.h
                                 include/DataFormatsITSMFT/NoiseMap.h
                                 include/DataFormatsITSMFT/TimeDeadMap.h
                                  include/DataFormatsITSMFT/Cluster.h
                                  include/DataFormatsITSMFT/CompCluster.h
                                  include/DataFormatsITSMFT/ClusterPattern.h
                                  include/DataFormatsITSMFT/ClusterTopology.h
                                  include/DataFormatsITSMFT/TopologyDictionary.h
                                  include/DataFormatsITSMFT/CTF.h
                                  include/DataFormatsITSMFT/TrkClusRef.h
                                  include/DataFormatsITSMFT/PhysTrigger.h
                          LINKDEF src/ITSMFTDataFormatsLinkDef.h)

o2_add_test(Cluster
            SOURCES test/test_Cluster.cxx
            COMPONENT_NAME DataFormatsITSMFT
            PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT)
