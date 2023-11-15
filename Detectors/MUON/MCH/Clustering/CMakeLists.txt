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

find_package(GSL REQUIRED)

o2_add_library(MCHClustering
               SOURCES src/ClusterOriginal.cxx
                       src/ClusterFinderOriginal.cxx
                       src/ClusterizerParam.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHMappingInterface O2::MCHBase O2::MCHPreClustering
                                     O2::Framework O2::CommonUtils)

o2_target_root_dictionary(MCHClustering
                          HEADERS include/MCHClustering/ClusterizerParam.h)

o2_add_library(MCHClusteringGEM
               SOURCES src/ClusterConfig.cxx
                       src/ClusterDump.cxx
                       src/ClusterFinderGEM.cxx
                       src/clusterProcessing.cxx
                       src/ClusterPEM.cxx
                       src/mathUtil.cxx
                       src/mathieson.cxx
                       src/mathiesonFit.cxx
                       src/PadsPEM.cxx
                       src/poissonEM.h
                       src/poissonEM.cxx
                       src/InspectModel.cxx
               PUBLIC_LINK_LIBRARIES GSL::gsl O2::MCHMappingInterface O2::MCHBase O2::MCHPreClustering O2::MCHClustering
                                     O2::Framework O2::CommonUtils)

