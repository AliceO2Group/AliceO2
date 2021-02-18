# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsTRD
               SOURCES src/TriggerRecord.cxx
                       src/LinkRecord.cxx
                       src/Tracklet64.cxx
                       src/RawData.cxx
                       PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::SimulationDataFormat)

           o2_target_root_dictionary(DataFormatsTRD
               HEADERS include/DataFormatsTRD/TriggerRecord.h
                       include/DataFormatsTRD/LinkRecord.h
                       include/DataFormatsTRD/Tracklet64.h
                       include/DataFormatsTRD/RawData.h
                       include/DataFormatsTRD/Constants.h
                       include/DataFormatsTRD/CalibratedTracklet.h
                       include/DataFormatsTRD/Hit.h)
