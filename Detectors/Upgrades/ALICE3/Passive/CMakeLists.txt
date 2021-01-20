# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(Alice3DetectorsPassive
               SOURCES src/Pipe.cxx
                       src/PassiveBase.cxx
               PUBLIC_LINK_LIBRARIES O2::Field O2::DetectorsBase O2::SimConfig)

o2_target_root_dictionary(Alice3DetectorsPassive
                          HEADERS include/Alice3DetectorsPassive/Pipe.h
                                  include/Alice3DetectorsPassive/PassiveBase.h
                          LINKDEF src/PassiveLinkDef.h)