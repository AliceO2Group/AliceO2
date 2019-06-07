o2_add_test_root_macro(DisplayEvents.C
                       PUBLIC_LINK_LIBRARIES
                       EventVisualisationView
                       ITSMFTReconstruction
                       ITSBase
                       DataFormatsITS
                       LABELS "its")

o2_add_test_root_macro(simple_geom_ITS.C
                       PUBLIC_LINK_LIBRARIES
                       EventVisualisationView
                       ITSMFTReconstruction
                       ITSBase
                       DataFormatsITS
                       LABELS "its")
