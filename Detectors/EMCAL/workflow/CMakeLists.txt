# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(EMCALWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/PublisherSpec.cxx
                       src/DigitsPrinterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsEMCAL
                                     O2::DPLUtils O2::EMCALBase O2::Algorithm)

o2_add_executable(reco-workflow
                  COMPONENT_NAME emcal
                  SOURCES src/emc-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::EMCALWorkflow)
