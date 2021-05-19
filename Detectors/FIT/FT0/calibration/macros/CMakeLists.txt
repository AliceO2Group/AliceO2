o2_add_test_root_macro(
  makeDummyFT0CalibObjectInCCDB.C
  PUBLIC_LINK_LIBRARIES
    O2::DataFormatsFT0
    O2::Framework
    O2::CCDB
    O2::FT0Calibration
    O2::DetectorsCalibration)

install(FILES makeDummyFT0CalibObjectInCCDB.C DESTINATION share/macro/)
