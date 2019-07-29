# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_executable(reco-workflow
                  SOURCES src/mid-reco-workflow.cxx
                  COMPONENT_NAME mid
                  SOURCES src/mid-reco-workflow.cxx
                          src/ClusterizerSpec.cxx
                          src/ClusterLabelerSpec.cxx
                          src/DigitReaderSpec.cxx
                          src/RecoWorkflow.cxx
                          src/TrackerSpec.cxx
                          src/TrackLabelerSpec.cxx
                          TARGETVARNAME
                          exename
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::SimConfig
                                        ms_gsl::ms_gsl
                                        O2::SimulationDataFormat
                                        O2::DataFormatsMID
                                        O2::DPLUtils
                                        O2::MIDSimulation
                                        O2::MIDTracking)

target_include_directories(
  ${exename}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
