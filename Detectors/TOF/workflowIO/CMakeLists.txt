# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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
                       src/TOFMatchedReaderSpec.cxx
                       src/CalibInfoReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::TOFBase O2::DataFormatsTOF O2::TOFReconstruction)

