# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TOFCalibration
               SOURCES src/CalibTOFapi.cxx
	               src/CalibTOF.cxx
		       src/CollectCalibInfoTOF.cxx
                       src/LHCClockCalibrator.cxx
		       src/TOFChannelCalibrator.cxx
		       src/TOFCalibCollector.cxx
		       src/TOFDCSProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTOF O2::TOFBase
                                     O2::CCDB
	                             O2::DetectorsCalibration
	                             O2::DetectorsDCS
				     ROOT::Minuit
	                             ms_gsl::ms_gsl)
				  
				   
o2_target_root_dictionary(TOFCalibration
                          HEADERS include/TOFCalibration/CalibTOFapi.h
                                  include/TOFCalibration/CalibTOF.h
                                  include/TOFCalibration/LHCClockCalibrator.h
                                  include/TOFCalibration/TOFChannelCalibrator.h
                                  include/TOFCalibration/TOFCalibCollector.h
                                  include/TOFCalibration/CollectCalibInfoTOF.h
				  include/TOFCalibration/TOFDCSProcessor.h)


o2_add_executable(data-generator-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/data-generator-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
	                                O2::DataFormatsTOF
					O2::TOFBase)

o2_add_executable(lhc-clockphase-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/lhc-clockphase-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                        O2::TOFCalibration
	                                O2::DetectorsCalibration)

o2_add_executable(tof-channel-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-channel-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                        O2::TOFCalibration
	                                O2::DetectorsCalibration)

o2_add_executable(tof-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                        O2::TOFCalibration
	                                O2::DetectorsCalibration)


o2_add_executable(tof-dummy-ccdb-for-calib
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-dummy-ccdb-for-calib.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                        O2::TOFCalibration
	                                O2::DetectorsCalibration)

o2_add_executable(tof-collect-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-collect-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
 		                        O2::TOFCalibration
	                                O2::DetectorsCalibration)

o2_add_executable(tof-dcs-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-dcs-data-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                       O2::TOFCalibration
                                       O2::DetectorsDCS)

o2_add_executable(tof-dcs-sim-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/tof-dcs-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

