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

o2_add_library(HMPIDReconstruction
               TARGETVARNAME targetName
               SOURCES src/Clusterer.cxx
                       src/HmpidDecoder2.cxx
                       src/HmpidEquipment.cxx
                       src/CTFHelper.cxx
                       src/Recon.cxx
                       src/CTFCoder.cxx
               PUBLIC_LINK_LIBRARIES O2::HMPIDBase
                                     O2::DataFormatsHMP
                                     ROOT::Physics
                                     O2::HMPIDSimulation)

o2_target_root_dictionary(HMPIDReconstruction
                          HEADERS include/HMPIDReconstruction/Clusterer.h
                                  include/HMPIDReconstruction/HmpidDecoder2.h
                                  include/HMPIDReconstruction/Recon.h
                                  include/HMPIDReconstruction/HmpidEquipment.h)
