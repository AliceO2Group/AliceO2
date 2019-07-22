# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsFITT0
               SOURCES src/Digit.cxx src/RecPoints.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsFITT0
                          HEADERS include/DataFormatsFITT0/Digit.h
                                  include/DataFormatsFITT0/RecPoints.h
                                  include/DataFormatsFITT0/MCLabel.h
                                  include/DataFormatsFITT0/HitType.h)
