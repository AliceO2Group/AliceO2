# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EC0Workflow
               SOURCES src/RecoWorkflow.cxx
                       src/DigitReaderSpec.cxx
                       src/ClustererSpec.cxx
                       src/ClusterWriterSpec.cxx
                       src/TrackerSpec.cxx
                       src/CookedTrackerSpec.cxx
                       src/TrackWriterSpec.cxx
                       src/TrackReaderSpec.cxx
		       src/VertexReaderSpec.cxx
	       PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::DataFormatsITS
                                     O2::SimulationDataFormat
                                     O2::EC0tracking
                                     O2::EC0Reconstruction
                                     O2::EndCapsReconstruction
                                     O2::EndCapsWorkflow
                                     O2::GPUTracking)

o2_add_executable(reco-workflow
                  SOURCES src/ec0-reco-workflow.cxx
                  COMPONENT_NAME ec0
                  PUBLIC_LINK_LIBRARIES O2::EC0Workflow)
