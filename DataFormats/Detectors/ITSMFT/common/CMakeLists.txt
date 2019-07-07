# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsITSMFT
               SOURCES src/ROFRecord.cxx
                       src/Cluster.cxx
                       src/CompCluster.cxx
                       src/ClusterPattern.cxx
                       src/ClusterTopology.cxx
                       src/TopologyDictionary.cxx
               PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats)

o2_target_root_dictionary(DataFormatsITSMFT
                          HEADERS include/DataFormatsITSMFT/ROFRecord.h
                                  include/DataFormatsITSMFT/Cluster.h
                                  include/DataFormatsITSMFT/CompCluster.h
                                  include/DataFormatsITSMFT/ClusterPattern.h
                                  include/DataFormatsITSMFT/ClusterTopology.h
                                  include/DataFormatsITSMFT/TopologyDictionary.h
                          LINKDEF src/ITSMFTDataFormatsLinkDef.h)
