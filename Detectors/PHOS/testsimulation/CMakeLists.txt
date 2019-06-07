o2_add_test_root_macro(PutPhosInTop.C PUBLIC_LINK_LIBRARIES DetectorsPassive
                       FairRoot::Base PHOSSimulation)

o2_add_test_root_macro(drawPHOSgeometry.C PUBLIC_LINK_LIBRARIES DetectorsPassive
                       FairRoot::Base PHOSSimulation)

o2_add_test_root_macro(plot_clu_phos.C
                       PUBLIC_LINK_LIBRARIES
                       FairRoot::Base
                       SimulationDataFormat
                       PHOSReconstruction
                       PHOSBase)

o2_add_test_root_macro(plot_dig_phos.C
                       PUBLIC_LINK_LIBRARIES
                       FairRoot::Base
                       SimulationDataFormat
                       PHOSSimulation
                       PHOSBase)

o2_add_test_root_macro(plot_hit_phos.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       SimulationDataFormat PHOSSimulation)

o2_add_test_root_macro(run_clu_phos.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       PHOSReconstruction)

o2_add_test_root_macro(run_digi_phos.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       PHOSSimulation)

o2_add_test_root_macro(run_sim_phos.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       DetectorsPassive PHOSSimulation)
