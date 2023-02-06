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


# MCHWorkflow library is (at least) needed by Detectors/CTF/workflow
o2_add_library(MCHWorkflow
               SOURCES
                   src/ClusterFinderOriginalSpec.cxx
                   src/ClusterFinderGEMSpec.cxx
                   src/ClusterReaderSpec.cxx
                   src/ClusterWriterSpec.cxx
                   src/DataDecoderSpec.cxx
                   src/PreClusterReaderSpec.cxx
                   src/PreClusterWriterSpec.cxx
                   src/TrackReaderSpec.cxx
                   src/TrackTreeReader.cxx
                   src/TrackWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES
                   O2::CommonUtils
                   O2::DPLUtils
                   O2::DataFormatsParameters
                   O2::SimulationDataFormat
                   O2::DetectorsRaw
                   O2::MCHCTF
                   O2::MCHClustering
                   O2::MCHClusteringGEM
                   O2::MCHPreClustering
                   O2::MCHRawCommon
                   O2::MCHRawDecoder
                   ROOT::TreePlayer
               )

o2_add_executable(
        cru-page-reader-workflow
        SOURCES src/cru-page-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsRaw O2::DPLUtils Boost::program_options)

o2_add_executable(
        dump-pages-workflow
        SOURCES src/dump-pages-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsRaw O2::DPLUtils Boost::program_options)

# to be deprecated : use DevIO/digits-writer instead
o2_add_executable(
        digits-sink-workflow
        SOURCES src/digits-sink-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils Boost::program_options O2::MCHBase O2::MCHRawDecoder)

o2_add_executable(
        raw-to-digits-workflow
        SOURCES src/raw-to-digits-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow O2::MCHMappingImpl4)

o2_add_executable(
        raw-debug-workflow
        SOURCES src/raw-debug-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        cru-page-to-digits-workflow
        SOURCES src/cru-page-to-digits-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        event-finder-workflow
        SOURCES src/EventFinderSpec.cxx src/event-finder-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::MCHTriggering)

o2_add_executable(
        digits-reader-workflow
        SOURCES src/DigitSamplerSpec.cxx src/digits-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsRaw O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3)

o2_add_executable(
        sim-digits-reader-workflow
        SOURCES src/DigitReaderSpec.cxx src/sim-digits-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsRaw O2::DPLUtils O2::DataFormatsMCH O2::SimulationDataFormat)

o2_add_executable(
        preclusters-sink-workflow
        SOURCES src/PreClusterSinkSpec.cxx src/preclusters-sink-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3)

o2_add_executable(
        preclusters-writer-workflow
        SOURCES src/preclusters-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        preclusters-reader-workflow
        SOURCES src/preclusters-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        preclusters-to-clusters-original-workflow
        SOURCES src/preclusters-to-clusters-original-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow O2::MCHClustering O2::MCHMappingImpl4)

o2_add_executable(
        preclusters-to-clusters-gem-workflow
        SOURCES src/preclusters-to-clusters-GEM-workflow.cxx src/ClusterFinderGEMSpec.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHClusteringGEM O2::MCHMappingImpl3)

o2_add_executable(
        clusters-sink-workflow
        SOURCES src/clusters-sink-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3)

o2_add_executable(
        clusters-sampler-workflow
        SOURCES src/clusters-sampler-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase)

o2_add_executable(
        clusters-writer-workflow
        SOURCES src/clusters-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        clusters-reader-workflow
        SOURCES src/clusters-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        vertex-sampler-workflow
        SOURCES src/VertexSamplerSpec.cxx src/vertex-sampler-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH)

o2_add_executable(
        tracks-to-tracks-at-vertex-workflow
        SOURCES src/TrackAtVertexSpec.cxx src/tracks-to-tracks-at-vertex-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::DataFormatsMCH O2::MCHTracking)

o2_add_executable(
        tracks-sink-workflow
        SOURCES src/TrackSinkSpec.cxx src/tracks-sink-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase)

o2_add_executable(
        tracks-writer-workflow
        SOURCES src/tracks-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        tracks-reader-workflow
        SOURCES src/tracks-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        tracks-sampler-workflow
        SOURCES src/TrackSamplerSpec.cxx src/tracks-sampler-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase)

o2_add_executable(
        entropy-encoder-workflow
        SOURCES src/entropy-encoder-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        reco-workflow
        SOURCES
            src/DigitReaderSpec.cxx
            src/EventFinderSpec.cxx
            src/TrackMCLabelFinderSpec.cxx
            src/reco-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
            O2::MCHDigitFiltering
            O2::MCHGeometryTransformer
            O2::MCHMappingImpl4
            O2::MCHTimeClustering
            O2::MCHTriggering
            O2::MCHTracking
            O2::MCHWorkflow
            O2::SimulationDataFormat
            O2::Steer
        TARGETVARNAME mch-reco-workflow)

o2_add_executable(tracks-file-dumper
        SOURCES src/tracks-file-dumper.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES Boost::program_options O2::MCHWorkflow)

o2_add_executable(
        tracks-mc-label-finder-workflow
        SOURCES src/TrackMCLabelFinderSpec.cxx src/tracks-mc-label-finder-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::Steer O2::DataFormatsMCH O2::MCHBase)

o2_add_executable(rofs-histogrammer
        SOURCES src/rofs-histogrammer.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH)
