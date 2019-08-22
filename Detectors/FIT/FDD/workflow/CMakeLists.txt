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
               SOURCES src/RecoWorkflow.cxx src/DigitReaderSpec.cxx
                       src/ReconstructorSpec.cxx src/RecPointWriterSpec.cxx
                       src/RecPointReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::FDDReconstruction O2::Framework)

o2_add_executable(reco-workflow
                  COMPONENT_NAME fdd
                  SOURCES src/fdd-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FDDWorkflow)
