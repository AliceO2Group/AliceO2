o2_add_test_root_macro(drawCPVgeometry.C PUBLIC_LINK_LIBRARIES CPVSimulation
                       DetectorsPassive PHOSSimulation)

o2_add_test_root_macro(plot_hit_cpv.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       CPVSimulation)

o2_add_test_root_macro(plot_dig_cpv.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       CPVSimulation)

o2_add_test_root_macro(run_digi_cpv.C PUBLIC_LINK_LIBRARIES FairRoot::Base
                       CPVSimulation)

o2_add_test_root_macro(run_sim_cpv.C
                       PUBLIC_LINK_LIBRARIES
                       FairRoot::Base
                       DetectorsPassive
                       CPVSimulation
                       Generators)
