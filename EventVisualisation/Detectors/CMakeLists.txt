# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EventVisualisationDetectors
               SOURCES
                    src/DataInterpreterTPC.cxx
                    src/DataReaderTPC.cxx
                    src/DataInterpreterITS.cxx
                    src/DataReaderITS.cxx
                    src/DataInterpreterVSD.cxx
                    src/DataReaderVSD.cxx
               PUBLIC_LINK_LIBRARIES
                       O2::EventVisualisationBase 
                       O2::DataFormatsTPC
                       O2::DataFormatsITSMFT
                       O2::DataFormatsITS
                       O2::ITSBase
                       O2::TPCBase
)
