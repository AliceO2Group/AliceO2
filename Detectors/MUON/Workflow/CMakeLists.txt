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

o2_add_executable(
        tracks-matcher-workflow
        SOURCES src/TrackMatcherSpec.cxx src/TrackWriterSpec.cxx src/tracks-matcher-workflow.cxx
        COMPONENT_NAME muon
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::MCHWorkflow O2::MIDWorkflow O2::MUONMatching)

o2_add_executable(
        tracks-writer-workflow
        SOURCES src/TrackWriterSpec.cxx src/tracks-writer-workflow.cxx
        COMPONENT_NAME muon
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::ReconstructionDataFormats)
