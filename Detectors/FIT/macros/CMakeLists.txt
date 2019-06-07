o2_add_test_root_macro(readHitsDigits.C PUBLIC_LINK_LIBRARIES DataFormatsFITT0)

o2_add_test_root_macro(run_reco_t0.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base
                                             FairLogger::FairLogger
                                             T0Reconstruction)
