o2_add_test_root_macro(
  makeChannelTimeOffsetFV0CalibObjectInCCDB.C
  PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                        O2::FV0Simulation
                        O2::DataFormatsFV0
                        O2::Framework
                        O2::CCDB
                        O2::FV0Calibration
                        O2::DetectorsCalibration
)

o2_add_test_root_macro(
  readChannelTimeOffsetFV0CalibObjectFromCCDB.C
  PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                        O2::FV0Simulation
                        O2::DataFormatsFV0
                        O2::Framework
                        O2::CCDB
                        O2::FV0Calibration
                        O2::DetectorsCalibration
)

install(
  FILES makeChannelTimeOffsetFV0CalibObjectInCCDB.C 
        readChannelTimeOffsetFV0CalibObjectFromCCDB.C
  DESTINATION share/macro/
)
