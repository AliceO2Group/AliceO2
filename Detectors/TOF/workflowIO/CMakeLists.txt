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

o2_add_library(TOFWorkflowIO
               SOURCES src/DigitReaderSpec.cxx
                       src/ClusterReaderSpec.cxx
                       src/TOFClusterWriterSpec.cxx
                       src/TOFDigitWriterSpec.cxx
                       src/TOFRawWriterSpec.cxx
                       src/TOFCalClusInfoWriterSpec.cxx
                       src/CalibClusReaderSpec.cxx
                       src/TOFMatchedWriterSpec.cxx
                       src/TOFCalibWriterSpec.cxx
                       src/TOFMatchableWriterSpec.cxx
                       src/TOFMatchedReaderSpec.cxx
                       src/CalibInfoReaderSpec.cxx
                       src/TOFIntegrateClusterWriterSpec.cxx
                       src/TOFIntegrateClusterReaderSpec.cxx
                       src/CcdbRequestSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::TOFBase O2::DataFormatsTOF O2::TOFReconstruction O2::Algorithm)

o2_add_executable(test-ccdb-request
                  SOURCES src/test-ccdb-request.cxx
                  COMPONENT_NAME tof
                  PUBLIC_LINK_LIBRARIES O2::TOFWorkflowIO)
