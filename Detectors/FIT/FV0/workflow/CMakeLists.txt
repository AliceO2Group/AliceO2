# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FV0Workflow
               SOURCES src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/DigitReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsFV0
                                     O2::FV0Reconstruction
                                     O2::DetectorsCommonDataFormats
                                     O2::DPLUtils)

o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow)

o2_add_executable(digit-reader-workflow
                  SOURCES src/digits-reader-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow)

o2_add_executable(flp-dpl-workflow
                  COMPONENT_NAME fv0
                  SOURCES src/fv0-flp-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow O2::FITWorkflow O2::FV0Raw
                  TARGETVARNAME fv0flpexe)

if(NOT APPLE)
 set_property(TARGET ${fitrecoexe} PROPERTY LINK_WHAT_YOU_USE ON)
endif()    		
