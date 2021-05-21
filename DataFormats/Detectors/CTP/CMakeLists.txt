# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsCTP
  SOURCES src/Digits.cxx 
  SOURCES src/Configuration.cxx
  PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                        O2::Headers
                        O2::SimulationDataFormat
                        O2::CommonConstants
                        O2::DataFormatsFT0)
o2_target_root_dictionary(DataFormatsCTP
                          HEADERS include/DataFormatsCTP/Digits.h
                                  include/DataFormatsCTP/Configuration.h)

