# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FDDWorkflow
               SOURCES src/DigitReaderSpec.cxx
	               src/DigitWriterSpec.cxx
	               src/EntropyEncoderSpec.cxx
		       src/EntropyDecoderSpec.cxx
	       	       src/RecoWorkflow.cxx 
                       src/ReconstructorSpec.cxx 
		       src/RecPointWriterSpec.cxx
                       src/RecPointReaderSpec.cxx

                       src/RawDataProcessSpec.cxx
                       src/RawDataReaderSpec.cxx
                       src/RawReaderFDD.cxx
		       src/RawWorkflow.cxx
		       
               PUBLIC_LINK_LIBRARIES O2::DataFormatsFDD
	       			     O2::FDDReconstruction 
	       			     O2::Framework 
				     O2::DPLUtils
				     O2::FDDRaw)

o2_add_executable(reco-workflow
                  COMPONENT_NAME fdd
                  SOURCES src/fdd-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FDDWorkflow)
		  
o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME fdd
                  PUBLIC_LINK_LIBRARIES O2::FDDWorkflow)

o2_add_executable(digit-reader-workflow
                  SOURCES src/digits-reader-workflow.cxx
                  COMPONENT_NAME fdd
                  PUBLIC_LINK_LIBRARIES O2::FDDWorkflow)

o2_add_executable(flp-dpl-workflow
                  COMPONENT_NAME fdd
                  SOURCES src/fdd-flp-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FDDWorkflow
                  TARGETVARNAME fddflpexe)
