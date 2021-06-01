# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHClustering
               SOURCES src/ClusterOriginal.cxx
                       src/ClusterFinderOriginal.cxx
                       src/MathiesonOriginal.cxx
                       src/ClusterizerParam.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHMappingInterface O2::MCHBase O2::MCHPreClustering
                                     O2::Framework O2::CommonUtils)

o2_target_root_dictionary(MCHClustering
                          HEADERS include/MCHClustering/ClusterizerParam.h)
