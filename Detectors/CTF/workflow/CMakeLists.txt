# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CTFWorkflow
               SOURCES src/CTFWriterSpec.cxx
                       src/CTFReaderSpec.cxx
	       PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DetectorsCommonDataFormats
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsTPC
                                     O2::DataFormatsTOF
                                     O2::DataFormatsFT0
                                     O2::DataFormatsFV0
				     O2::DataFormatsFDD
				     O2::DataFormatsMID
				     O2::DataFormatsPHOS
				     O2::DataFormatsCPV
                                     O2::DataFormatsParameters
                                     O2::ITSMFTWorkflow
                                     O2::TPCWorkflow
                                     O2::FT0Workflow
                                     O2::FV0Workflow
				     O2::FDDWorkflow
                                     O2::TOFWorkflow
                                     O2::MIDWorkflow
                                     O2::EMCALWorkflow
				     O2::PHOSWorkflow
				     O2::CPVWorkflow
                                     O2::Algorithm
                                     O2::CommonUtils)

o2_add_executable(writer-workflow
                  SOURCES src/ctf-writer-workflow.cxx
                  COMPONENT_NAME ctf
                  PUBLIC_LINK_LIBRARIES O2::CTFWorkflow)

o2_add_executable(reader-workflow
                  SOURCES src/ctf-reader-workflow.cxx
                  COMPONENT_NAME ctf
                  PUBLIC_LINK_LIBRARIES O2::CTFWorkflow)

