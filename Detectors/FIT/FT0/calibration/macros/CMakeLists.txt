o2_add_test_root_macro(
        makeChannelOffsetCalibObjectInCCDB.C
        PUBLIC_LINK_LIBRARIES O2::DataFormatsFT0 O2::Framework O2::CCDB O2::DetectorsCalibration
        LABELS ft0)

o2_add_test_root_macro(
        makeFT0CCDBEntryForDCS.C
        PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::CCDB
        LABELS ft0)

o2_add_test_root_macro(
        readFT0DCSentries.C
        PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::CCDB)