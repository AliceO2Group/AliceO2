# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EMCALCalibration
               SOURCES src/EMCALChannelCalibrator.cxx 
	       PUBLIC_LINK_LIBRARIES O2::CCDB O2::EMCALBase
	                             O2::DetectorsCalibration
	                             O2::DataFormatsEMCAL
				                 O2::Framework
				     	         O2::Algorithm)
				  
				   
o2_target_root_dictionary(EMCALCalibration
                          HEADERS include/EMCALCalibration/EMCALChannelCalibrator.h
                          LINKDEF src/EMCALCalibrationLinkDef.h)



o2_add_executable(emcal-channel-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/emc-channel-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                                O2::EMCALCalibration
	                                    O2::DetectorsCalibration)

