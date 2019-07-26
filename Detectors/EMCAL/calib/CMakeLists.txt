# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EMCALCalib
               SOURCES src/BadChannelMap.cxx
                       src/TimeCalibrationParams.cxx
                       src/TempCalibrationParams.cxx
                       src/CalibDB.cxx
	       PUBLIC_LINK_LIBRARIES O2::CCDB O2::EMCALBase)

o2_target_root_dictionary(EMCALCalib
                          HEADERS include/EMCALCalib/BadChannelMap.h
                                  include/EMCALCalib/TimeCalibrationParams.h
                                  include/EMCALCalib/TempCalibrationParams.h
                                  include/EMCALCalib/CalibDB.h
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
            LABELS emcal)

o2_add_test(TempCalibrationParams
            SOURCES test/testTempCalibration.cxx
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

o2_add_test_root_macro(macros/TempCalibrationParams_CCDBApiTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/TempCalibrationParams_CalibDBTest.C
            PUBLIC_LINK_LIBRARIES O2::EMCALCalib
            LABELS emcal COMPILE_ONLY)

o2_data_file(COPY files DESTINATION Detectors/EMCAL)
