o2_add_test_root_macro(PutEmcalInTop.C PUBLIC_LINK_LIBRARIES EMCALSimulation
                       DetectorsPassive FairRoot::Base)

o2_add_test_root_macro(drawEMCALgeometry.C PUBLIC_LINK_LIBRARIES EMCALSimulation
                       DetectorsPassive FairRoot::Base)

o2_add_test_root_macro(run_sim_emcal.C
                       PUBLIC_LINK_LIBRARIES
                       FairRoot::Base
                       EMCALSimulation
                       DetectorsPassive
                       Generators)
