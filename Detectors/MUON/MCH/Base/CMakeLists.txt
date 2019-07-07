# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHBase
               SOURCES src/Mapping.cxx src/DigitBlock.cxx
                       src/PreClusterBlock.cxx src/ClusterBlock.cxx
                       src/TrackBlock.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Core FairRoot::Base FairMQ::FairMQ)

o2_target_root_dictionary(MCHBase
                          HEADERS include/MCHBase/Mapping.h
                                  include/MCHBase/DigitBlock.h
                                  include/MCHBase/PreClusterBlock.h
                                  include/MCHBase/ClusterBlock.h
                                  include/MCHBase/TrackBlock.h)
