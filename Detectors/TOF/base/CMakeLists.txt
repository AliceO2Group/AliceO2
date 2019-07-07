# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TOFBase
               SOURCES src/Geo.cxx src/Digit.cxx
               PUBLIC_LINK_LIBRARIES Boost::serialization FairRoot::Base
                                     O2::DetectorsBase)

o2_target_root_dictionary(TOFBase
                          HEADERS include/TOFBase/Geo.h include/TOFBase/Digit.h)

o2_add_test(TOFIndex
            SOURCES test/testTOFIndex.cxx
            COMPONENT_NAME TOF
            PUBLIC_LINK_LIBRARIES O2::TOFBase)
