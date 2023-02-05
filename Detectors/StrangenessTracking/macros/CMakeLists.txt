o2_add_test_root_macro(XiTrackingStudy.C
                       PUBLIC_LINK_LIBRARIES O2::MathUtils
                                             O2::ITSBase
                                             O2::ITSMFTReconstruction
                                             O2::ITSMFTSimulation
                                             O2::DataFormatsITSMFT
                                             O2::DataFormatsITS
                                             O2::SimulationDataFormat
                                             O2::StrangenessTracking
                                             O2::DCAFitter
                       LABELS strangeness-tracking)
