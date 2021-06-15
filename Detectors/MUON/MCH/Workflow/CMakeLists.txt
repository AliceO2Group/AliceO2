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
                   src/EntropyDecoderSpec.cxx
                   src/DataDecoderSpec.cxx
                   src/PreClusterFinderSpec.cxx
                   src/ClusterFinderOriginalSpec.cxx
               PUBLIC_LINK_LIBRARIES
                   O2::CommonUtils
                   O2::DPLUtils
                   O2::DataFormatsParameters
                   O2::DetectorsRaw
                   O2::MCHCTF
                   O2::MCHClustering
                   O2::MCHPreClustering
                   O2::MCHRawCommon
                   O2::MCHRawDecoder
               )

o2_add_executable(
        cru-page-reader-workflow
        SOURCES src/cru-page-reader-workflow.cxx
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
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

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
        digits-to-preclusters-workflow
        SOURCES src/digits-to-preclusters-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        digits-reader-workflow
        SOURCES src/DigitSamplerSpec.cxx src/digits-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsRaw O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3)

o2_add_executable(
        sim-digits-reader-workflow
        SOURCES src/DigitReaderSpec.cxx src/sim-digits-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3 O2::SimulationDataFormat)


o2_add_executable(
        preclusters-sink-workflow
        SOURCES src/PreClusterSinkSpec.cxx src/preclusters-sink-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase O2::MCHMappingImpl3)

o2_add_executable(
        preclusters-to-clusters-original-workflow
        SOURCES src/preclusters-to-clusters-original-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

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
        clusters-to-tracks-original-workflow
        SOURCES src/TrackFinderOriginalSpec.cxx src/clusters-to-tracks-original-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHTracking)

o2_add_executable(
        clusters-to-tracks-workflow
        SOURCES src/TrackFinderSpec.cxx src/clusters-to-tracks-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::DataFormatsParameters O2::Framework O2::DataFormatsMCH O2::MCHTracking O2::DataFormatsParameters)

o2_add_executable(
        clusters-transformer-workflow
        SOURCES src/clusters-transformer-workflow.cxx src/ClusterTransformerSpec.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
           O2::DetectorsBase
           O2::Framework
           O2::MCHBase
           O2::MCHGeometryTransformer
        )

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
        tracks-sampler-workflow
        SOURCES src/TrackSamplerSpec.cxx src/tracks-sampler-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHBase)

o2_add_executable(
        tracks-to-tracks-workflow
        SOURCES src/TrackFitterSpec.cxx src/tracks-to-tracks-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH O2::MCHTracking)

o2_add_executable(
        entropy-encoder-workflow
        SOURCES src/entropy-encoder-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHWorkflow)

o2_add_executable(
        reco-workflow
        SOURCES
            src/ClusterTransformerSpec.cxx
            src/TrackAtVertexSpec.cxx
            src/TrackFinderSpec.cxx
            src/VertexSamplerSpec.cxx
            src/reco-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
            O2::MCHGeometryTransformer
            O2::MCHTracking
            O2::MCHWorkflow
        )

o2_add_executable(tracks-file-dumper
  SOURCES src/tracks-file-dumper.cxx
  COMPONENT_NAME mch
  PUBLIC_LINK_LIBRARIES fmt::fmt Boost::program_options O2::MCHBase)
