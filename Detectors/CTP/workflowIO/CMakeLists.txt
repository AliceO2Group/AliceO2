# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CTPWorkflowIO
               SOURCES src/DigitReaderSpec.cxx
                       src/DigitWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsCTP
                                     O2::DPLUtils
                                     O2::DetectorsRaw
                                     O2::Algorithm)

o2_add_executable(digit-writer
                  COMPONENT_NAME ctp
                  SOURCES src/digits-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CTPWorkflowIO)
