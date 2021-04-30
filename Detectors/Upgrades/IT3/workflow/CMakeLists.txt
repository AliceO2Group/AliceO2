# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITS3Workflow
               SOURCES src/DigitReaderSpec.cxx
                       src/DigitWriterSpec.cxx
                       src/RecoWorkflow.cxx
                       src/ClusterWriterWorkflow.cxx
                       src/ClustererSpec.cxx
                       src/ClusterWriterSpec.cxx
                      #  src/TrackerSpec.cxx
                      #  src/CookedTrackerSpec.cxx
                      #  src/TrackWriterSpec.cxx
                      #  src/TrackReaderSpec.cxx
		      #  src/VertexReaderSpec.cxx
	       PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::DataFormatsITS
                                     O2::SimulationDataFormat
                                     O2::ITS3Simulation
                                    #  O2::ITStracking
                                    #  O2::ITSReconstruction
                                     O2::ITS3Reconstruction
                                    #  O2::ITSMFTWorkflow
                                     O2::GPUTracking
)

o2_add_executable(digit-writer-workflow
                  SOURCES src/digit-writer-workflow.cxx
                  COMPONENT_NAME its3
                  PUBLIC_LINK_LIBRARIES O2::ITS3Workflow)

o2_add_executable(digit-reader-workflow
                  SOURCES src/digit-reader-workflow.cxx
                  COMPONENT_NAME its3
                  PUBLIC_LINK_LIBRARIES O2::ITS3Workflow)

o2_add_executable(reco-workflow
                  SOURCES src/its3-reco-workflow.cxx
                  COMPONENT_NAME its3
                  PUBLIC_LINK_LIBRARIES O2::ITS3Workflow)
       
# o2_add_executable(cluster-writer-workflow
#                   SOURCES src/its-cluster-writer-workflow.cxx
#                   COMPONENT_NAME its
#                   PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

# o2_add_executable(cluster-reader-workflow
#                   SOURCES src/its-cluster-reader-workflow.cxx
#                   COMPONENT_NAME its
#                   PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

