# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(EventVisualisationDetectors
               SOURCES
                    src/DataInterpreterITS.cxx
                    src/DataInterpreterTPC.cxx
                    src/DataInterpreterVSD.cxx
                    src/DataReaderITS.cxx
                    src/DataReaderJSON.cxx
                    src/DataReaderTPC.cxx
                    src/DataReaderVSD.cxx
               PUBLIC_LINK_LIBRARIES
                       O2::EventVisualisationBase 
                       O2::DataFormatsTPC
                       O2::DataFormatsITSMFT
                       O2::DataFormatsITS
                       O2::ITSBase
                       O2::TPCBase
)
