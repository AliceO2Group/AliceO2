# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsFV0
        SOURCES src/Hit.cxx
                src/ChannelData.cxx
                src/BCData.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                              O2::CommonDataFormat
                              ms_gsl::ms_gsl)

o2_target_root_dictionary(DataFormatsFV0
        HEADERS include/DataFormatsFV0/Hit.h
                include/DataFormatsFV0/BCData.h
                include/DataFormatsFV0/ChannelData.h)
