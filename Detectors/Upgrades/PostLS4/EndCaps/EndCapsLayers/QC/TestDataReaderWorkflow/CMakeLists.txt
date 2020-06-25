# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITSQCDataReaderWorkflow
               SOURCES src/TestDataReaderWorkflow.cxx
                       src/TestDataReader.cxx
                       src/TestDataGetter.cxx
                       PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::DataFormatsITS
                                     O2::SimulationDataFormat
                                     O2::ITStracking
                                     O2::ITSReconstruction)

install(DIRECTORY include/ITSQCDataReaderWorkflow DESTINATION include)

o2_add_executable(qc-workflow
                  SOURCES src/its-qc-data-reader.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSQCDataReaderWorkflow)
