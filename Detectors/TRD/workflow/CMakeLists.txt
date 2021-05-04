# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

#add_compile_options(-O0 -pg -fPIC)

add_subdirectory(io)

o2_add_library(TRDWorkflow
               TARGETVARNAME targetName
               SOURCES src/TRDDigitizerSpec.cxx
                       src/TRDTrapSimulatorSpec.cxx
                       src/TRDTrackletTransformerSpec.cxx
                       src/TRDGlobalTrackingSpec.cxx
                       src/TRDTrackWriterSpec.cxx
                       src/TRDTrackReaderSpec.cxx
                       src/TRDTrackingWorkflow.cxx
                       src/EntropyDecoderSpec.cxx
                       src/EntropyEncoderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::Steer O2::Algorithm O2::DataFormatsTRD O2::TRDSimulation O2::TRDReconstruction O2::DetectorsBase O2::SimulationDataFormat O2::TRDBase O2::TRDCalibration O2::GPUTracking O2::GlobalTrackingWorkflowReaders O2::ReconstructionDataFormats O2::TPCWorkflow O2::TRDWorkflowIO)

                   #o2_target_root_dictionary(TRDWorkflow
                   # HEADERS include/TRDWorkflow/TRDTrapSimulatorSpec.h)

o2_add_executable(trap-sim
                  COMPONENT_NAME trd
                  SOURCES src/TRDTrapSimulatorWorkFlow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::DPLUtils
                                        O2::Steer
                                        O2::TRDBase
                                        O2::DataFormatsTRD
                                        O2::TRDWorkflow)

o2_add_executable(global-tracking
                  COMPONENT_NAME trd
                  SOURCES src/trd-tracking-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(tracklet-transformer
                  COMPONENT_NAME trd
                  SOURCES src/TRDTrackletTransformerWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(track-reader
                  COMPONENT_NAME trd
                  SOURCES src/trd-track-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)



if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
