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

add_subdirectory(macros)

o2_add_library(TRDReconstruction
               SOURCES src/CTFCoder.cxx
                       src/CTFHelper.cxx
                       src/CruRawReader.cxx
                       src/DataReaderTask.cxx
                       src/EventRecord.cxx
               PUBLIC_LINK_LIBRARIES O2::TRDBase
                                     O2::DataFormatsTRD
                                     O2::DataFormatsTPC
                                     O2::TRDWorkflowIO
                                     O2::ReconstructionDataFormats
                                     O2::DetectorsRaw
                                     O2::rANS
                                     O2::DataFormatsCTP
                                     Microsoft.GSL::GSL)


o2_add_executable(datareader
    COMPONENT_NAME trd
    SOURCES src/DataReader.cxx
    PUBLIC_LINK_LIBRARIES O2::TRDReconstruction
    )
