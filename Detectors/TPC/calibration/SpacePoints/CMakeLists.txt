# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(SpacePoints
               SOURCES src/Param.cxx src/TrackResiduals.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC)

o2_target_root_dictionary(SpacePoints
                          HEADERS include/SpacePoints/Param.h
                                  include/SpacePoints/TrackResiduals.h
                          LINKDEF src/SpacePointCalibLinkDef.h)
