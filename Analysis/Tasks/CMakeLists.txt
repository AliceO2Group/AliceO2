# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_dpl_workflow(trackselection
                  SOURCES trackselection.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(correlations
                  SOURCES correlations.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(correlations-collection
                  SOURCES correlationsCollection.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(correlations-mixed
                  SOURCES correlationsMixed.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(vertexing-hf
	  	  SOURCES vertexerhf.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsVertexing
	          COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hftrackindexskimscreator
                  SOURCES hftrackindexskimscreator.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsVertexing
	          COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hfcandidatecreator2prong
                  SOURCES hfcandidatecreator2prong.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsVertexing
	          COMPONENT_NAME Analysis)

o2_add_dpl_workflow(taskdzero
                  SOURCES taskdzero.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsVertexing
	          COMPONENT_NAME Analysis)

o2_add_dpl_workflow(validation
	  	  SOURCES validation.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase
	          COMPONENT_NAME Analysis)

if(FastJet_FOUND)
o2_add_dpl_workflow(jetfinder
  SOURCES jetfinder.cxx
  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                        FastJet::FastJet
  COMPONENT_NAME Analysis)
endif()

o2_add_dpl_workflow(upc-an
                  SOURCES upcAnalysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(event-selection
                  SOURCES eventSelection.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::CCDB
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(event-selection-qa
                  SOURCES eventSelectionQa.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(multiplicity-table
                  SOURCES multiplicityTable.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(multiplicity-qa
                  SOURCES multiplicityQa.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase
                  COMPONENT_NAME Analysis)
                  
o2_add_dpl_workflow(centrality-table
                  SOURCES centralityTable.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::CCDB
                  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(centrality-qa
                  SOURCES centralityQa.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase
                  COMPONENT_NAME Analysis)
                  
o2_add_dpl_workflow(invmass-an
                  SOURCES invMassAnalysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase
                  COMPONENT_NAME Analysis)
                  
o2_add_dpl_workflow(table-maker
                  SOURCES tableMaker.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                  COMPONENT_NAME Analysis)                                    
                  
o2_add_dpl_workflow(table-reader
                  SOURCES tableReader.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                  COMPONENT_NAME Analysis)                          

o2_add_dpl_workflow(timestamp
                    SOURCES timestamp.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::DetectorsRaw O2::AnalysisCore O2::CommonDataFormat O2::CCDB
                    COMPONENT_NAME Analysis)
