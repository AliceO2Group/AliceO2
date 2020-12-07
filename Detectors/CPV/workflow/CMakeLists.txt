# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(CPVWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/PublisherSpec.cxx
                       src/ClusterizerSpec.cxx
                       src/DigitsPrinterSpec.cxx
                       src/RawToDigitConverterSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework 
                                     O2::DataFormatsCPV 
                                     O2::DPLUtils 
                                     O2::CPVBase 
                                     O2::CPVCalib 
                                     O2::CPVSimulation 
                                     O2::CPVReconstruction 
                                     O2::Algorithm)

o2_add_executable(reco-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/cpv-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CPVWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CPVWorkflow)
