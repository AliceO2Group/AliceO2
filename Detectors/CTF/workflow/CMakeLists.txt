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

o2_add_library(CTFWorkflow
               SOURCES src/CTFWriterSpec.cxx
                       src/CTFReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DetectorsCommonDataFormats
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsTPC
                                     O2::DataFormatsTRD
                                     O2::DataFormatsTOF
                                     O2::DataFormatsFT0
                                     O2::DataFormatsFV0
                                     O2::DataFormatsFDD
                                     O2::DataFormatsMID
                                     O2::DataFormatsPHOS
                                     O2::DataFormatsCPV
                                     O2::DataFormatsZDC
                                     O2::DataFormatsHMP
                                     O2::DataFormatsCTP
                                     O2::DataFormatsParameters
                                     O2::ITSMFTWorkflow
                                     O2::TOFWorkflowUtils
                                     O2::TPCWorkflow
                                     O2::TRDWorkflow
                                     O2::FT0Workflow
                                     O2::FV0Workflow
                                     O2::FDDWorkflow
                                     O2::MCHCTF
                                     O2::MIDWorkflow
                                     O2::EMCALWorkflow
                                     O2::PHOSWorkflow
                                     O2::CPVWorkflow
                                     O2::ZDCWorkflow
                                     O2::HMPIDWorkflow
                                     O2::CTPWorkflow
                                     O2::Algorithm
                                     O2::CommonUtils)

o2_add_executable(writer-workflow
                  SOURCES src/ctf-writer-workflow.cxx
                  COMPONENT_NAME ctf
                  PUBLIC_LINK_LIBRARIES O2::CTFWorkflow)

o2_add_executable(reader-workflow
                  SOURCES src/ctf-reader-workflow.cxx
                  COMPONENT_NAME ctf
                  TARGETVARNAME targetName
                  PUBLIC_LINK_LIBRARIES O2::CTFWorkflow)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
