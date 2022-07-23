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

o2_add_library(EventVisualisationDataConverter
               SOURCES src/VisualisationEvent.cxx
                       src/VisualisationTrack.cxx
                       src/VisualisationCluster.cxx
                       src/VisualisationCalo.cxx
                       src/VisualisationEventSerializer.cxx
                       src/VisualisationEventJSONSerializer.cxx
                       src/VisualisationEventROOTSerializer.cxx
               PUBLIC_LINK_LIBRARIES RapidJSON::RapidJSON
                        O2::ReconstructionDataFormats
)

o2_add_executable(eve-convert
        SOURCES src/converter.cxx
                src/VisualisationEvent.cxx
                src/VisualisationEventSerializer.cxx
                src/VisualisationEventJSONSerializer.cxx
                src/VisualisationEventROOTSerializer.cxx
                src/VisualisationTrack.cxx
                src/VisualisationCluster.cxx
                src/VisualisationCalo.cxx
        PUBLIC_LINK_LIBRARIES
                O2::EventVisualisationView
                RapidJSON::RapidJSON
                O2::ReconstructionDataFormats
        )
