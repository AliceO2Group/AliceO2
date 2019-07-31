# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FT0Base
               SOURCES src/Geometry.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Physics FairRoot::Base)

o2_target_root_dictionary(FT0Base HEADERS include/FT0Base/Geometry.h)

o2_data_file(COPY files DESTINATION Detectors/FT0/)
