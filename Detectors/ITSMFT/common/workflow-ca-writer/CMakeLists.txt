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

o2_add_library(ITSMFTCAWriter
               SOURCES src/ITSCATrackWriterSpec.cxx
                       src/MFTCATrackWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimulationDataFormat
                                     O2::DataFormatsITS
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsMFT
                                     O2::MFTTracking)

o2_add_test(itsmft-ca-writer-contract
            COMPONENT_NAME itsmft
            LABELS "itsmft;workflow"
            SOURCES test/testITSMFTCAWriterContract.cxx
            PUBLIC_LINK_LIBRARIES O2::ITSMFTCAWriter)
