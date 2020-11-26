# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
		  
o2_add_library(FemtoDream
               SOURCES  src/FemtoDreamContainer.cxx
                        src/FemtoDreamCollisionSelection.cxx
                        src/FemtoDreamTrackSelection.cxx
                        src/FemtoDreamV0Cuts.cxx
                        src/FemtoDreamCascadeCuts.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore)

o2_target_root_dictionary(FemtoDream
				HEADERS include/o2femtodream/FemtoDreamContainer.h
						include/o2femtodream/FemtoDreamCollisionSelection.h
						include/o2femtodream/FemtoDreamTrackSelection.h
						include/o2femtodream/FemtoDreamV0Cuts.h
						include/o2femtodream/FemtoDreamCascadeCuts.h
				LINKDEF src/FemtoDreamLinkDef.h)
		  
o2_add_dpl_workflow(femtoDreamTaskTrackProducer
					SOURCES femtoDreamTaskTrackProducer.cxx
					PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel O2::FemtoDream
					COMPONENT_NAME Analysis)
					
		  
o2_add_dpl_workflow(femtoDreamTaskSameEvent
					SOURCES femtoDreamTaskSameEvent.cxx
					PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel O2::FemtoDream
					COMPONENT_NAME Analysis)
					
o2_add_dpl_workflow(femtoDreamTaskMixedEvent
					SOURCES femtoDreamTaskMixedEvent.cxx
					PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel O2::FemtoDream
					COMPONENT_NAME Analysis)