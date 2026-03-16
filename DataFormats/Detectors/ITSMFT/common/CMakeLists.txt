# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
                       src/TimeDeadMap.cxx
                       src/CTF.cxx
                       src/DPLAlpideParam.cxx
                       src/DPLAlpideParamInitializer.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                       O2::DetectorsCommonDataFormats
                       O2::ReconstructionDataFormats
                       O2::CommonUtils
                       Microsoft.GSL::GSL)

o2_target_root_dictionary(DataFormatsITSMFT
                          HEADERS include/DataFormatsITSMFT/ROFRecord.h
                                  include/DataFormatsITSMFT/Digit.h
                                  include/DataFormatsITSMFT/DPLAlpideParam.h
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
