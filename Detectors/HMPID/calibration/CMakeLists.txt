# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(HMPIDCalibration
		TARGETVARNAME targetName
                SOURCES src/HMPIDDCSProcessor.cxx 
			#src/HMPIDTestComp.cxx 
			src/HMPIDDCSTime.cxx
#                       src/HmpidDecoder2.cxx
#                       src/HmpidEquipment.cxx
#                       src/CTFHelper.cxx
#                       src/CTFCoder.cxx
                PUBLIC_LINK_LIBRARIES O2::HMPIDBase
                                      O2::DataFormatsHMP
                                      O2::HMPIDSimulation
                                      O2::SimulationDataFormat
                                      O2::DetectorsBase
				      O2::DetectorsCalibration
				      O2::DetectorsDCS
                                      O2::DetectorsRaw
                                      O2::CCDB
                                      ROOT::Physics
				      ROOT::Minuit
				      O2::Steer
				      Microsoft.GSL::GSL)


o2_target_root_dictionary(HMPIDCalibration
                          HEADERS  include/HMPIDCalibration/HMPIDDCSProcessor.h
                            	   include/HMPIDCalibration/HMPIDDCSTime.h)
                            	   #include/HMPIDCalibration/HMPIDTestComp.h)
#                                  include/HMPIDCalibration/HmpidDecoder2.h
#                                  include/HMPIDCalibration/HmpidEquipment.h)
#                                  include/HMPIDSimulation/Detector.h
#                                  include/HMPIDSimulation/HMPIDDigitizer.h)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_add_executable(dcs-sim-workflow # run with o2-hmpid-dcs-sim-workflow
                   COMPONENT_NAME hmpid # adds hmpid to the above command
                   SOURCES testWorkflow/hmpid-dcs-sim-workflow.cxx
                   PUBLIC_LINK_LIBRARIES O2::Framework
					 O2::DCStestWorkflow)



 
