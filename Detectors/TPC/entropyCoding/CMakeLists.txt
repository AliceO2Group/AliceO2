# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCEntropyCoding
               SOURCES src/TPCEntropyEncoder.cxx
                       src/TPCEntropyDecoder.cxx
               PUBLIC_LINK_LIBRARIES O2::rANS O2::DataFormatsTPC 
                                     O2::Framework ROOT::Tree ROOT::RIO)

o2_target_root_dictionary(TPCEntropyCoding
                          HEADERS include/TPCEntropyCoding/EncodedClusters.h)

o2_add_executable(rans-encoder-standalone
                  COMPONENT_NAME tpc
                  SOURCES run/encoder-standalone.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCEntropyCoding
                                        O2::DataFormatsTPC  ROOT::RIO
                                        O2::Framework)
                          
o2_add_executable(rans-decoder-standalone
                  COMPONENT_NAME tpc
                  SOURCES run/decoder-standalone.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCEntropyCoding O2::DataFormatsTPC ROOT::RIO)
