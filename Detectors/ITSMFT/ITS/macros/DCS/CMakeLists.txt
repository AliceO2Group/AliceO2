o2_add_test_root_macro(
  makeITSCCDBEntryForDCS.C
  PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::CCDB)

install(
  FILES makeITSCCDBEntryForDCS.C
  DESTINATION share/macro/)
