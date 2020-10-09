# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsDCS
               TARGETVARNAME targetName
               SOURCES src/Clock.cxx
		       src/DataPointCompositeObject.cxx
		       src/DataPointIdentifier.cxx
		       src/DataPointValue.cxx
		       src/DeliveryType.cxx
		       src/GenericFunctions.cxx
		       src/StringUtils.cxx
		       src/DCSProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::Headers
	                             O2::CommonUtils
				     O2::CCDB
				     O2::DetectorsCalibration
	                             ms_gsl::ms_gsl)

o2_target_root_dictionary(DetectorsDCS
                          HEADERS include/DetectorsDCS/DataPointCompositeObject.h
			          include/DetectorsDCS/DataPointIdentifier.h
				  include/DetectorsDCS/DataPointValue.h
				  include/DetectorsDCS/DCSProcessor.h)

o2_add_executable(dcs-data-workflow
		  COMPONENT_NAME dcs
		  SOURCES testWorkflow/dcs-data-workflow.cxx
		  PUBLIC_LINK_LIBRARIES O2::Framework
					O2::DetectorsDCS)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
