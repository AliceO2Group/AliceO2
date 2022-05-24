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

o2_add_library(EMCALCalib
               SOURCES src/BadChannelMap.cxx
                       src/TimeCalibrationParams.cxx
                       src/TimeCalibParamL1Phase.cxx
                       src/TempCalibrationParams.cxx
                       src/TempCalibParamSM.cxx
                       src/GainCalibrationFactors.cxx
                       src/TriggerTRUDCS.cxx
                       src/TriggerSTUDCS.cxx
                       src/TriggerSTUErrorCounter.cxx
                       src/TriggerDCS.cxx
                       src/FeeDCS.cxx
                       src/CalibDB.cxx
                       src/ElmbMeasurement.cxx
               PUBLIC_LINK_LIBRARIES O2::CCDB O2::EMCALBase)

o2_target_root_dictionary(EMCALCalib
                          HEADERS include/EMCALCalib/BadChannelMap.h
                                  include/EMCALCalib/TimeCalibrationParams.h
                                  include/EMCALCalib/TimeCalibParamL1Phase.h
                                  include/EMCALCalib/TempCalibrationParams.h
                                  include/EMCALCalib/TempCalibParamSM.h
                                  include/EMCALCalib/GainCalibrationFactors.h
                                  include/EMCALCalib/TriggerTRUDCS.h
                                  include/EMCALCalib/TriggerSTUDCS.h
                                  include/EMCALCalib/TriggerSTUErrorCounter.h
                                  include/EMCALCalib/TriggerDCS.h
                                  include/EMCALCalib/FeeDCS.h
                                  include/EMCALCalib/CalibDB.h
                                  include/EMCALCalib/ElmbData.h
                                  include/EMCALCalib/ElmbMeasurement.h
                          LINKDEF src/EMCALCalibLinkDef.h)

o2_add_test(BadChannelMap
            SOURCES test/testBadChannelMap.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal)

o2_add_test(TimeCalibrationParams
            SOURCES test/testTimeCalibration.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(TimeCalibParamL1Phase
            SOURCES test/testTimeL1PhaseCalib.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(TempCalibrationParams
            SOURCES test/testTempCalibration.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(TempCalibParamSM
            SOURCES test/testTempCalibrationSM.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(GainCalibrationFactors
            SOURCES test/testGainCalibration.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(TriggerTRUDCS
            SOURCES test/testTriggerTRUDCS.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal)

o2_add_test(TriggerSTUDCS
            SOURCES test/testTriggerSTUDCS.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal)

o2_add_test(TriggerSTUErrorCounter
            SOURCES test/testTriggerSTUErrorCounter.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal)

o2_add_test(TriggerDCS
            SOURCES test/testTriggerDCS.cxx
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            COMPONENT_NAME emcal
            LABELS emcal)

o2_add_test_root_macro(macros/BadChannelMap_CCDBApitest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/BadChannelMap_CalibDBtest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TimeCalibrationParams_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TimeCalibrationParams_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TimeCalibParamsL1Phase_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TimeCalibParamsL1Phase_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TempCalibrationParams_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TempCalibrationParams_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TempCalibParamSM_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TempCalibParamSM_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/GainCalibrationFactors_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/GainCalibrationFactors_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_data_file(COPY files DESTINATION Detectors/EMC)

