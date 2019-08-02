#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_test_root_macro(PutPhosInTop.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsPassive FairRoot::Base
                       O2::PHOSSimulation
                       LABELS phos)

o2_add_test_root_macro(drawPHOSgeometry.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsPassive FairRoot::Base
                       O2::PHOSSimulation
                       LABELS phos)

o2_add_test_root_macro(plot_dig_phos.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base
                       O2::SimulationDataFormat
                       O2::PHOSSimulation O2::PHOSBase
                       LABELS phos)

o2_add_test_root_macro(plot_hit_phos.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base
                       O2::SimulationDataFormat
                       O2::PHOSSimulation
                       LABELS phos)

o2_add_test_root_macro(run_digi_phos.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base O2::PHOSSimulation
                       LABELS phos)

o2_add_test_root_macro(run_sim_phos.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base O2::DetectorsPassive
                       O2::PHOSSimulation
                       LABELS phos)
