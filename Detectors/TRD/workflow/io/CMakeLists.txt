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

o2_add_library(TRDWorkflowIO
               SOURCES src/TRDDigitReaderSpec.cxx
                       src/TRDTrackletReaderSpec.cxx
                       src/TRDCalibReaderSpec.cxx
                       src/TRDDigitWriterSpec.cxx
                       src/TRDTrackletWriterSpec.cxx
                       src/TRDTrapRawWriterSpec.cxx
                       src/TRDCalibratedTrackletWriterSpec.cxx
                       src/TRDTrackWriterSpec.cxx
                       src/TRDTrackReaderSpec.cxx
                       src/TRDCalibWriterSpec.cxx
                       src/TRDPHReaderSpec.cxx
                       src/KrClusterWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD O2::SimulationDataFormat O2::DPLUtils O2::GPUDataTypeHeaders O2::DataFormatsTPC)


o2_add_executable(digit-reader-workflow
                 COMPONENT_NAME trd
                 SOURCES src/digit-reader-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::TRDWorkflowIO)
o2_add_executable(track-reader
                 COMPONENT_NAME trd
                 SOURCES src/trd-track-reader-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::TRDWorkflowIO)
o2_add_executable(digittracklet-writer
                 COMPONENT_NAME trd
                 SOURCES src/trd-digittracklet-writer-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::TRDWorkflowIO)
